r"""
epistemic_speculative_pager.py - Speculative Epistemic KV-Cache Paging & ELLA
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements:
  1. Epistemic Working Set Estimator (EWSE) with 5-tier piecewise ELLA Retention Metric (Section 4.1, Eq. 3):
       - System Anchor (P_anchor): 10^9 (excluded from Resident_evictable(HBM))
       - Active Window (P_active, W_local = 64 tokens = 4 pages for executing stream):
           \tilde{A}_i(P_k, t) = max(A_class,i(P_k, t), 10^7 * exp(-kappa * \Delta t_win))
        - Tool Schema (P_tool):
            5e8 * I_tool,i(P_k, t) + (1 - I_tool,i(P_k, t)) * (W_tool * exp(-lambda_tool * \Delta t_i) + C_floor + eta_DAG * \hat{d}_{DAG,i}(P_k, t)^{-1})
        - Conversation History (P_hist):
            max(C_hist, W_hist * exp(-lambda * \Delta t_i) * (1 - H_i(t) + epsilon))
            + B_DAG * I_DAG,i(P_k, t) + eta_DAG * \hat{d}_{DAG,i}(P_k, t)^{-1}
        - Scratchpad (P_scratch):
            W_scratch * exp(-mu * \Delta t_i) * H_i(t) + eta_DAG * \hat{d}_{DAG,i}(P_k, t)^{-1}
  2. Multi-tenant shared-page aggregation (Eq. 4):
       A(P_k, t) = max_{S_i in O(P_k)} \tilde{A}_i(P_k, t)
     with tie-breaking attributes (\hat{d}_{DAG,i^\dagger}^{-1}, i^\dagger, t_{last_access,i^\dagger}) inherited from
     the maximizing owner stream S_{i^\dagger(P_k, t)} in argmax_{S_i in O(P_k)} \tilde{A}_i(P_k, t).
  3. Restricted evictable victim domain (Invariant 1, Eq. 6):
       Resident_evictable(HBM) = Resident_clean(HBM) \ (P_anchor U P_pinned_prefetch U P_active_unsealed)
     with clean sealed write-through host mirroring and prefetch-pin release fallback prior to anchor eviction.
  4. Asynchronous DMA Speculative Prefetcher (ADSP) restricted to lookahead-predicted non-resident
     candidates (I_tool,i(P_k, t) = 1 or I_DAG,i(P_k, t) = 1 within d_look = 2 steps), evaluated causally at
     dispatch step t = n - \tau_lookahead.
"""

from __future__ import annotations
import math
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from paging_types import (
    SCORE_ANCHOR,
    SCORE_ACTIVE_TOOL,
    W_WIN,
    KAPPA_WIN,
    W_TOOL,
    LAMBDA_TOOL,
    C_FLOOR,
    C_HIST,
    W_HIST,
    EPSILON_HIST,
    LAMBDA_HIST,
    B_DAG_HIST,
    ETA_DAG,
    W_SCRATCH,
    MU_SCRATCH,
    W_LOCAL_STEPS,
    W_LOCAL_BLOCKS,
    D_LOOKAHEAD,
    TAU_LOOKAHEAD_STEPS,
    M_PRE_MAX,
    T_FAULT_PARTIAL_US,
    PageDescriptor,
    PageType,
    AgentEpistemicState,
    MemoryTier,
)
from memory_hierarchy import PhysicalMemoryHierarchy


class EpistemicSpeculativePager:
    """
    Hardware-aware speculative KV-cache manager with epistemic attention scoring (ELLA)
    and Asynchronous DMA Speculative Prefetching (ADSP).
    """

    def __init__(
        self,
        memory: PhysicalMemoryHierarchy,
        lookahead_steps: int = M_PRE_MAX,
        enable_speculative_prefetch: bool = True,
        use_epistemic_entropy: bool = True,
        write_through_enabled: bool = True,
        lambda_hist: float = LAMBDA_HIST,
        lambda_tool: float = LAMBDA_TOOL,
        mu_scratch: float = MU_SCRATCH,
        b_dag_hist: float = B_DAG_HIST,
    ):
        self.memory = memory
        self.lookahead_steps = min(lookahead_steps, M_PRE_MAX)
        self.enable_speculative_prefetch = enable_speculative_prefetch
        self.use_epistemic_entropy = use_epistemic_entropy
        self.write_through_enabled = write_through_enabled
        self.lambda_hist = lambda_hist
        self.lambda_tool = lambda_tool
        self.mu_scratch = mu_scratch
        self.b_dag_hist = b_dag_hist
        self._lock = threading.RLock()

        # Epistemic states per agent
        self._agent_states: Dict[str, AgentEpistemicState] = {}
        self._global_step: int = 0
        self._active_executing_agent_id: Optional[str] = None

        # Telemetry
        self.total_accesses = 0
        self.hits = 0
        self.speculative_hits = 0
        self.partial_overlap_hits = 0
        self.prefetches_dispatched = 0
        self.expired_prefetch_pins = 0
        self.misses = 0
        self.evictions = 0
        self.total_stall_ms = 0.0

    def register_agent_state(self, state: AgentEpistemicState):
        with self._lock:
            self._agent_states[state.agent_id] = state
            if self._active_executing_agent_id is None:
                self._active_executing_agent_id = state.agent_id

    def update_agent_state(
        self,
        agent_id: str,
        entropy: float,
        is_tool_pending: bool = False,
        tool_name: Optional[str] = None,
        is_history_summary_pending: bool = False,
    ):
        with self._lock:
            self._active_executing_agent_id = agent_id
            for aid, st_obj in self._agent_states.items():
                st_obj.is_active_executing = (aid == agent_id)
            if agent_id in self._agent_states:
                st = self._agent_states[agent_id]
                st.update_ema_entropy(entropy)
                st.is_tool_call_pending = is_tool_pending
                st.is_history_summary_pending = is_history_summary_pending
                st.target_tool_name = tool_name
                st.turn_depth += 1
                st.step_index = self._global_step

    def _get_inverse_dag_distance(
        self, page: PageDescriptor, agent_state: Optional[AgentEpistemicState]
    ) -> Tuple[int, float]:
        r"""
        Returns (\hat{d}_{DAG,i}(P_k, t), \hat{d}_{DAG,i}(P_k, t)^{-1} in [0, 1]) along the
        mode-selected branch (\arg\max_{v in V_act} \tilde{p}_{i,m}(v)).
        """
        d_hat = page.dag_distance_d_hat
        if agent_state:
            stream_d = page.per_stream_dag_distance.get(agent_state.agent_id, d_hat)
            d_hat = min(d_hat, stream_d)
            if page.page_type == PageType.TOOL_SCHEMA and agent_state.is_tool_call_pending:
                d_hat = 1
            elif page.page_type == PageType.CONVERSATION_HISTORY and agent_state.is_history_summary_pending:
                d_hat = 1
        inv_d = (1.0 / float(d_hat)) if (0 < d_hat < 90) else 0.0
        return d_hat, inv_d

    def _compute_single_owner_score(
        self, page: PageDescriptor, agent_state: Optional[AgentEpistemicState]
    ) -> float:
        r"""
        Evaluates the exact 5-tier piecewise ELLA retention score \tilde{A}_i(P_k, t) (Section IV-A, Eq. 3).
        \Delta t_i(P_k, t) is measured in execution steps (with turn-depth fallback if step == 0),
        so idle tool schemas decay below DAG-boosted history (470) within ~8 steps (~1 turn).
        """
        # Tier 1: System Anchor & Key Sinks -> Permanently pinned (10^9)
        if page.page_type == PageType.SYSTEM_ANCHOR or page.is_anchor:
            return SCORE_ANCHOR

        cur_step = self._global_step
        cur_turn = agent_state.turn_depth if agent_state else 0
        stream_id = agent_state.agent_id if agent_state else page.sequence_id
        last_step_i = page.per_stream_last_access_step.get(stream_id, page.last_access_step)

        if cur_step > 0 and last_step_i > 0:
            dt_i = float(max(0, cur_step - last_step_i))
        elif cur_turn > 0 or page.last_access_turn > 0:
            dt_i = float(max(0, cur_turn - page.last_access_turn))
        else:
            dt_i = float(max(0, cur_step - last_step_i))

        # In STATIC_DAG_ELLA ablation, H_i(t) is fixed to 0.0
        h_val = (
            (agent_state.epistemic_entropy_h if agent_state else 0.50)
            if self.use_epistemic_entropy
            else 0.0
        )

        d_hat, inv_d = self._get_inverse_dag_distance(page, agent_state)
        dag_indicator = 1.0 if d_hat <= D_LOOKAHEAD else 0.0
        tool_indicator = 1.0 if (page.page_type == PageType.TOOL_SCHEMA and d_hat <= D_LOOKAHEAD) else 0.0

        # Evaluate class-specific retention score A_class,i(P_k, t) (Eq. 3)
        if page.page_type == PageType.TOOL_SCHEMA:
            class_score = (
                SCORE_ACTIVE_TOOL * tool_indicator
                + (1.0 - tool_indicator) * (W_TOOL * math.exp(-self.lambda_tool * dt_i) + C_FLOOR + ETA_DAG * inv_d)
            )
        elif page.page_type == PageType.CONVERSATION_HISTORY:
            hist_decay = W_HIST * math.exp(-self.lambda_hist * dt_i) * (1.0 - h_val + EPSILON_HIST)
            class_score = (
                max(C_HIST, hist_decay)
                + self.b_dag_hist * dag_indicator
                + ETA_DAG * inv_d
            )
        elif page.page_type == PageType.INTERMEDIATE_THOUGHT:
            class_score = W_SCRATCH * math.exp(-self.mu_scratch * max(1.0, dt_i)) * h_val + ETA_DAG * inv_d
        elif page.page_type == PageType.WORKING_ATTENTION:
            dt_win = min(3.0, float(dt_i // 16))
            class_score = W_WIN * math.exp(-KAPPA_WIN * dt_win)
        else:
            class_score = C_FLOOR

        # Non-Demoting Sliding Active Window Rule (Section III-B & Eq. 3):
        # For the currently executing stream S_{i*(t)}, pages in W_local = 64 tokens (4 blocks)
        # receive max(A_class,i(P_k, t), 10^7 * exp(-kappa * \Delta t_win)).
        is_executing_stream = (
            agent_state is None
            or agent_state.is_active_executing
            or self._active_executing_agent_id in (None, agent_state.agent_id)
        )
        in_active_window = (
            page.page_type == PageType.WORKING_ATTENTION
            or (
                page.page_type != PageType.INTERMEDIATE_THOUGHT
                and last_step_i > 0
                and (cur_step - last_step_i) <= W_LOCAL_STEPS
                and is_executing_stream
            )
        )
        if in_active_window and is_executing_stream:
            dt_win = min(3.0, float(max(0.0, dt_i) // 16))
            active_score = W_WIN * math.exp(-KAPPA_WIN * dt_win)
            return max(class_score, active_score)

        return class_score

    def compute_score_and_maximizing_owner(
        self, page: PageDescriptor, agent_state: Optional[AgentEpistemicState]
    ) -> Tuple[float, float, str, int]:
        r"""
        Computes projected future attention mass A(P_k, t) under ELLA (Eq. 4):
          A(P_k, t) = max_{S_i in O(P_k)} \tilde{A}_i(P_k, t),
        and returns (score, inv_d_i^\dagger, owner_id_i^\dagger, last_access_step_i^\dagger)
        inherited from the maximizing owner stream S_{i^\dagger(P_k, t)}.
        """
        owner_states: List[Optional[AgentEpistemicState]] = []
        if page.shared_owners:
            for oid in sorted(page.shared_owners):
                owner_states.append(self._agent_states.get(oid))
        if agent_state is not None and agent_state not in owner_states:
            owner_states.append(agent_state)
        if not owner_states:
            owner_states.append(self._agent_states.get(page.sequence_id))

        best_tuple: Optional[Tuple[float, float, str, int]] = None
        for st in owner_states:
            sc = self._compute_single_owner_score(page, st)
            _, inv_d = self._get_inverse_dag_distance(page, st)
            oid = st.agent_id if st else page.sequence_id
            last_step = page.per_stream_last_access_step.get(oid, page.last_access_step)
            cand_tuple = (sc, inv_d, oid, last_step)
            if best_tuple is None or cand_tuple > best_tuple:
                best_tuple = cand_tuple
        assert best_tuple is not None
        return best_tuple

    def compute_projected_attention_score(
        self, page: PageDescriptor, agent_state: Optional[AgentEpistemicState]
    ) -> float:
        """
        Computes projected future attention mass A(P_k, t) under ELLA (Eq. 4):
          A(P_k, t) = max_{S_i in O(P_k)} \tilde{A}_i(P_k, t).
        """
        return self.compute_score_and_maximizing_owner(page, agent_state)[0]

    def _expire_stale_speculative_pins(self):
        r"""Expires speculative ADSP prefetch pins once step n completes (after \tau_lookahead steps)."""
        for desc in list(self.memory._page_descriptors.values()):
            if desc.is_pinned and not desc.is_anchor and desc.pin_expiry_step >= 0:
                if self._global_step >= desc.pin_expiry_step:
                    desc.is_pinned = False
                    desc.pin_expiry_step = -1
                    self.expired_prefetch_pins += 1

    def trigger_speculative_prefetch(self, agent_id: str, candidate_pages: List[PageDescriptor]):
        r"""
        ADSP Speculative Prefetcher (Section IV-C):
        Dispatched causally at current step t = n - \tau_lookahead (\tau_lookahead = 2 steps ahead of step n).
        Restricts candidates to non-resident lookahead-predicted pages (I_tool,i(P_k, t) = 1 or I_DAG,i(P_k, t) = 1
        within d_look = 2 steps, or active window pages), sorts candidates in descending order of A(P_cand, t)
        (breaking ties by larger \hat{d}_{DAG,i^\dagger}^{-1}, maximizing stream index, and lowest block_index),
        and admits up to M_pre(n) <= 4 pages whenever A(P_cand, t) > A(P_victim, t).
        """
        if not self.enable_speculative_prefetch:
            return

        with self._lock:
            self._expire_stale_speculative_pins()
            agent_state = self._agent_states.get(agent_id)

            # Enforce global cap \tau_lookahead * M_pre_max = 8 simultaneously pinned prefetch pages
            current_pinned_prefetch = sum(
                1 for d in list(self.memory._page_descriptors.values())
                if d.is_pinned and not d.is_anchor and d.pin_expiry_step >= 0 and self.memory.is_page_in_device(d.page_id)
            )
            max_global_pinned = TAU_LOOKAHEAD_STEPS * M_PRE_MAX
            available_pin_budget = max(0, min(self.lookahead_steps, max_global_pinned - current_pinned_prefetch))
            if available_pin_budget == 0:
                return

            scored_candidates = []
            for p in candidate_pages:
                if self.memory.is_page_in_device(p.page_id):
                    continue
                d_hat, inv_d = self._get_inverse_dag_distance(p, agent_state)
                is_lookahead_predicted = (
                    d_hat <= D_LOOKAHEAD
                    or p.page_type == PageType.WORKING_ATTENTION
                    or (agent_state and agent_state.is_tool_call_pending and p.page_type == PageType.TOOL_SCHEMA)
                    or (agent_state and agent_state.is_history_summary_pending and p.page_type == PageType.CONVERSATION_HISTORY)
                )
                if not is_lookahead_predicted:
                    continue
                score, max_inv_d, max_oid, _ = self.compute_score_and_maximizing_owner(p, agent_state)
                p.projected_attention_mass = score
                if score > C_FLOOR:
                    scored_candidates.append((score, max_inv_d, max_oid, -p.block_index, p))

            # Sort descending by score, then larger \hat{d}_DAG^{-1}, maximizing owner id, lowest physical page index
            scored_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
            pages_to_fetch = [item[4] for item in scored_candidates[:available_pin_budget]]

            admitted_pages: List[PageDescriptor] = []
            for cand in pages_to_fetch:
                used, total = self.memory.get_device_occupancy()
                if used >= total:
                    victim_id = self._select_ella_victim(allow_prefetch_release_fallback=False)
                    if not victim_id:
                        # Resident_evictable(HBM) == empty: suppress speculative prefetch admission (Invariant 1)
                        break
                    victim_desc = self.memory._page_descriptors[victim_id]
                    victim_score = victim_desc.projected_attention_mass
                    # Only preempt victim if A(P_cand, t) > A(P_victim, t) at dispatch step t = n - \tau_lookahead
                    if cand.projected_attention_mass <= victim_score:
                        continue
                    self.memory.evict_device_page_to_host(victim_id, write_through_enabled=self.write_through_enabled)
                    self.evictions += 1
                admitted_pages.append(cand)

            for p in admitted_pages:
                p.is_pinned = True
                p.pin_expiry_step = self._global_step + TAU_LOOKAHEAD_STEPS * 4
                self.prefetches_dispatched += 1
                self.memory.schedule_speculative_prefetch(p.page_id)

    def access_page(self, page_desc: PageDescriptor, agent_id: str) -> Tuple[bool, bool, float]:
        """
        Accesses a page for the current token generation step.
        If prefetched in time: recorded as speculative hit with 0 stall time.
        If in-flight prefetch is partially overlapped: incurs residual T_FAULT_PARTIAL_US (2.50 us).
        If unpredicted demand miss: incurs clean write-through fault T_FAULT_CLEAN_US (9.00 us)
        when write_through_enabled=True, or T_FAULT_DIRTY_US (12.00 us) when write_through_enabled=False.
        """
        page_id = page_desc.page_id

        with self._lock:
            self._global_step += 1
            self.total_accesses += 1
            self._active_executing_agent_id = agent_id
            self._expire_stale_speculative_pins()

            now = time.time()
            page_desc.last_access_time = now
            page_desc.last_access_step = self._global_step
            page_desc.per_stream_last_access_step[agent_id] = self._global_step
            agent_state = self._agent_states.get(agent_id)
            if agent_state:
                page_desc.last_access_turn = agent_state.turn_depth
            page_desc.access_count += 1
            page_desc.projected_attention_mass = self.compute_projected_attention_score(page_desc, agent_state)

            # Stage-1 lookup transaction
            self.memory.total_physical_transactions += 1

            if self.memory.is_page_in_device(page_id):
                self.hits += 1
                if page_desc.pin_expiry_step >= 0:
                    self.speculative_hits += 1
                    page_desc.is_pinned = False
                    page_desc.pin_expiry_step = -1
                return True, True, 0.0

            # Check if an asynchronous DMA prefetch is currently in-flight
            if page_id in self.memory._dma_completed:
                event = self.memory._dma_completed[page_id]
                event.wait(timeout=0.01)
                if self.memory.is_page_in_device(page_id):
                    self.hits += 1
                    self.speculative_hits += 1
                    page_desc.is_pinned = False
                    page_desc.pin_expiry_step = -1
                    stall_ms = (T_FAULT_PARTIAL_US / 1000.0) if not event.is_set() else 0.0
                    self.total_stall_ms += stall_ms
                    return True, True, stall_ms

            # Synchronous Cache Miss
            self.misses += 1
            used, total = self.memory.get_device_occupancy()
            if used >= total:
                victim_id = self._select_ella_victim(allow_prefetch_release_fallback=True)
                if victim_id:
                    self.memory.evict_device_page_to_host(victim_id, write_through_enabled=self.write_through_enabled)
                    self.evictions += 1

            ok, stall_ms = self.memory.demand_swap_to_device(page_id, clean_victim=self.write_through_enabled)
            self.total_stall_ms += stall_ms
            return ok, False, stall_ms

    def _select_ella_victim(self, allow_prefetch_release_fallback: bool = True) -> Optional[str]:
        r"""
        Selects victim page in Resident_evictable(HBM) with minimum ELLA score (Invariant 1, Eq. 6):
          Resident_evictable(HBM) = Resident_clean(HBM) \ (P_anchor U P_pinned_prefetch U P_active_unsealed)
        where clean sealed pages have is_sealed == True and is_mirrored_to_host == True.
        If Resident_evictable(HBM) == empty under extreme concurrency, first awaits/selects any in-flight
        write-through sealed non-anchor page, and if none is available, releases the lowest-scoring
        prefetch-pinned page prior to ever evicting any P_anchor page (10^9).
        Ties are broken deterministically by smaller \hat{d}_{DAG,i^\dagger}(P_k)^{-1}, maximizing owner
        stream index, oldest LRU timestamp, and lowest physical page index k.
        """
        all_descs = list(self.memory._page_descriptors.values())
        candidates = [
            desc for desc in all_descs
            if self.memory.is_page_in_device(desc.page_id)
            and desc.is_sealed
            and (desc.is_mirrored_to_host or not self.write_through_enabled)
            and not desc.is_pinned
            and not desc.is_anchor
            and desc.page_type != PageType.SYSTEM_ANCHOR
        ]
        if not candidates and allow_prefetch_release_fallback:
            # Invariant 1 fallback step 1: check sealed unpinned non-anchor pages awaiting write-through
            candidates = [
                desc for desc in all_descs
                if self.memory.is_page_in_device(desc.page_id)
                and desc.is_sealed
                and not desc.is_pinned
                and not desc.is_anchor
                and desc.page_type != PageType.SYSTEM_ANCHOR
            ]
        if not candidates and allow_prefetch_release_fallback:
            # Invariant 1 fallback step 2: release lowest-scoring prefetch-pinned page prior to any P_anchor
            candidates = [
                desc for desc in all_descs
                if self.memory.is_page_in_device(desc.page_id)
                and desc.is_sealed
                and not desc.is_anchor
                and desc.page_type != PageType.SYSTEM_ANCHOR
            ]
        if not candidates:
            return None

        scored_victims = []
        for c in candidates:
            agent_st = self._agent_states.get(c.sequence_id)
            score, max_inv_d, max_oid, last_step = self.compute_score_and_maximizing_owner(c, agent_st)
            c.projected_attention_mass = score
            scored_victims.append((score, max_inv_d, max_oid, last_step, c.last_access_time, c.block_index, c.page_id))

        # Evict lowest A(P_j, t), then smallest \hat{d}_DAG^{-1}, owner id, oldest LRU timestamp, lowest block_index
        scored_victims.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
        return scored_victims[0][6]

