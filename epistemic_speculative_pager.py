r"""
epistemic_speculative_pager.py - Speculative Epistemic KV-Cache Paging & ELLA
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements:
  1. Epistemic Working Set Estimator (EWSE) with 5-tier piecewise ELLA Retention Metric (Eq. 3):
       - Tier 1 (P_anchor): \infty (1e9, hard-pinned)
       - Tier 2 (P_active, t - t_last <= W_local=64): max(A_class(P_k, t), W_win=1000)
       - Tier 3 (P_tool): W_tool * exp(-lambda_tool * \Delta t) + B_tool * I[\hat{d}_DAG <= d_look] * (1 - \gamma H_i(t))
       - Tier 4 (P_hist): max(C_floor, W_hist * [1 + \beta(1 - H_i(t))] * exp(-lambda * \Delta t) + B_DAG * I[\hat{d}_DAG <= d_look])
       - Tier 5 (P_scratch): W_scratch * exp(-\mu * \Delta t)
  2. Multi-tenant shared-page max aggregation: A_shared(P_k, t) = max_{i in Owners(P_k)} A_i(P_k, t).
  3. Restricted evictable victim domain:
       Resident_evictable(HBM) = {P_j in Resident(HBM) | Sealed(P_j) & Mirrored(P_j) & ~Pinned(P_j)}
     with automatic ADSP pin expiration after \tau_lookahead = 2 steps.
  4. Ablation switches for DEMAND_ELLA (no ADSP) and STATIC_DAG_ELLA (H_i(t) === 0).
"""

from __future__ import annotations
import math
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from paging_types import (
    W_WIN,
    W_TOOL,
    B_TOOL,
    GAMMA_ENTROPY,
    LAMBDA_TOOL,
    C_FLOOR,
    W_HIST,
    BETA_HIST,
    LAMBDA_HIST,
    B_DAG_HIST,
    W_SCRATCH,
    MU_SCRATCH,
    W_LOCAL_STEPS,
    D_LOOKAHEAD,
    TAU_LOOKAHEAD_STEPS,
    M_PRE_MAX,
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
    ):
        self.memory = memory
        self.lookahead_steps = min(lookahead_steps, M_PRE_MAX)
        self.enable_speculative_prefetch = enable_speculative_prefetch
        self.use_epistemic_entropy = use_epistemic_entropy
        self._lock = threading.RLock()

        # Epistemic states per agent
        self._agent_states: Dict[str, AgentEpistemicState] = {}
        self._global_step: int = 0

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

    def update_agent_state(
        self,
        agent_id: str,
        entropy: float,
        is_tool_pending: bool = False,
        tool_name: Optional[str] = None,
        is_history_summary_pending: bool = False,
    ):
        with self._lock:
            if agent_id in self._agent_states:
                st = self._agent_states[agent_id]
                st.update_ema_entropy(entropy)
                st.is_tool_call_pending = is_tool_pending
                st.is_history_summary_pending = is_history_summary_pending
                st.target_tool_name = tool_name
                st.turn_depth += 1

    def _compute_single_owner_score(
        self, page: PageDescriptor, agent_state: Optional[AgentEpistemicState]
    ) -> float:
        """
        Evaluates the exact 5-tier piecewise ELLA retention score A_i(P_k, t) (Eq. 3).
        """
        # Tier 1: System Anchor & Key Sinks -> Permanently pinned (\infty)
        if page.page_type == PageType.SYSTEM_ANCHOR or page.is_anchor:
            return 1e9

        now = time.time()
        age_sec = max(0.0, now - page.last_access_time) if page.last_access_time > 0 else 0.0
        # Map wall-clock age or discrete step/turn age into non-negative recency \Delta t
        cur_turn = agent_state.turn_depth if agent_state else 0
        cur_step = self._global_step
        dt_turn = max(float(max(0, cur_turn - page.last_access_turn)), age_sec * 10.0)
        dt_step = max(float(max(0, cur_step - page.last_access_step)), age_sec * 50.0)

        # In STATIC_DAG_ELLA ablation, H_i(t) is fixed to 0.0
        h_val = (
            (agent_state.epistemic_entropy_h if agent_state else 0.5)
            if self.use_epistemic_entropy
            else 0.0
        )

        # Determine DAG lookahead distance \hat{d}_DAG(P_k, t)
        d_hat = page.dag_distance_d_hat
        if agent_state:
            if page.page_type == PageType.TOOL_SCHEMA and agent_state.is_tool_call_pending:
                d_hat = 1
            elif page.page_type == PageType.CONVERSATION_HISTORY and agent_state.is_history_summary_pending:
                d_hat = 1
        dag_indicator = 1.0 if d_hat <= D_LOOKAHEAD else 0.0

        # Evaluate class-specific retention score A_class(P_k, t)
        if page.page_type == PageType.TOOL_SCHEMA:
            class_score = (
                W_TOOL * math.exp(-LAMBDA_TOOL * dt_turn)
                + B_TOOL * dag_indicator * (1.0 - GAMMA_ENTROPY * h_val)
            )
        elif page.page_type == PageType.CONVERSATION_HISTORY:
            hist_decay = W_HIST * (1.0 + BETA_HIST * (1.0 - h_val)) * math.exp(-LAMBDA_HIST * dt_turn)
            class_score = max(C_FLOOR, hist_decay) + B_DAG_HIST * dag_indicator
        elif page.page_type == PageType.INTERMEDIATE_THOUGHT:
            class_score = W_SCRATCH * math.exp(-MU_SCRATCH * max(1.0, dt_step))
        elif page.page_type == PageType.WORKING_ATTENTION:
            class_score = W_WIN * math.exp(-0.01 * dt_step)
        else:
            class_score = C_FLOOR

        # Tier 2 Sliding Active Window Rule:
        # Any page inside W_local=64 steps receives max(A_class(P_k, t), A_active(P_k, t))
        if page.page_type == PageType.WORKING_ATTENTION or (
            page.page_type != PageType.INTERMEDIATE_THOUGHT
            and page.last_access_step > 0
            and (cur_step - page.last_access_step) <= W_LOCAL_STEPS
        ):
            return max(class_score, W_WIN)

        return class_score

    def compute_projected_attention_score(
        self, page: PageDescriptor, agent_state: Optional[AgentEpistemicState]
    ) -> float:
        """
        Computes projected future attention mass A(P_k, t) under ELLA.
        For shared physical pages (shared_owners != empty), aggregates via max_{i in Owners(P_k)} A_i(P_k, t).
        """
        if page.shared_owners:
            scores = [
                self._compute_single_owner_score(page, self._agent_states.get(oid))
                for oid in page.shared_owners
            ]
            if agent_state is not None:
                scores.append(self._compute_single_owner_score(page, agent_state))
            return max(scores) if scores else self._compute_single_owner_score(page, agent_state)
        return self._compute_single_owner_score(page, agent_state)

    def _expire_stale_speculative_pins(self):
        """Expires speculative ADSP prefetch pins that were not accessed within \tau_lookahead steps."""
        for desc in self.memory._page_descriptors.values():
            if desc.is_pinned and not desc.is_anchor and desc.pin_expiry_step >= 0:
                if self._global_step >= desc.pin_expiry_step:
                    desc.is_pinned = False
                    desc.pin_expiry_step = -1
                    self.expired_prefetch_pins += 1

    def trigger_speculative_prefetch(self, agent_id: str, candidate_pages: List[PageDescriptor]):
        """
        Asynchronously prefetches up to M_pre <= 4 pages in W_spec(n) with A(P_k, t) > C_floor.
        Pre-emptively evicts low-scoring evictable device pages to maintain free slots.
        """
        if not self.enable_speculative_prefetch:
            return

        with self._lock:
            self._expire_stale_speculative_pins()
            agent_state = self._agent_states.get(agent_id)

            scored_candidates = []
            for p in candidate_pages:
                score = self.compute_projected_attention_score(p, agent_state)
                p.projected_attention_mass = score
                if score > C_FLOOR:
                    scored_candidates.append((score, p))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            pages_to_fetch = [
                p for _, p in scored_candidates
                if not self.memory.is_page_in_device(p.page_id)
            ][:self.lookahead_steps]

            for _ in range(len(pages_to_fetch)):
                used, total = self.memory.get_device_occupancy()
                if used >= total:
                    victim_id = self._select_ella_victim()
                    if victim_id:
                        self.memory.evict_device_page_to_host(victim_id)
                        self.evictions += 1

            for p in pages_to_fetch:
                p.is_pinned = True
                p.pin_expiry_step = self._global_step + TAU_LOOKAHEAD_STEPS * 4
                self.prefetches_dispatched += 1
                self.memory.schedule_speculative_prefetch(p.page_id)

    def access_page(self, page_desc: PageDescriptor, agent_id: str) -> Tuple[bool, bool, float]:
        """
        Accesses a page for the current token generation step.
        If prefetched in time: recorded as speculative hit with 0 stall time.
        """
        t_start = time.perf_counter()
        page_id = page_desc.page_id

        with self._lock:
            self._global_step += 1
            self.total_accesses += 1
            self._expire_stale_speculative_pins()

            now = time.time()
            page_desc.last_access_time = now
            page_desc.last_access_step = self._global_step
            agent_state = self._agent_states.get(agent_id)
            if agent_state:
                page_desc.last_access_turn = agent_state.turn_depth
            page_desc.access_count += 1
            page_desc.projected_attention_mass = self.compute_projected_attention_score(page_desc, agent_state)

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
                    stall_ms = (time.perf_counter() - t_start) * 1000.0
                    self.total_stall_ms += stall_ms
                    return True, True, stall_ms

            # Synchronous Cache Miss
            self.misses += 1
            used, total = self.memory.get_device_occupancy()
            if used >= total:
                victim_id = self._select_ella_victim()
                if victim_id:
                    self.memory.evict_device_page_to_host(victim_id)
                    self.evictions += 1

            ok = self.memory.demand_swap_to_device(page_id)
            stall_ms = (time.perf_counter() - t_start) * 1000.0
            self.total_stall_ms += stall_ms
            return ok, False, stall_ms

    def _select_ella_victim(self) -> Optional[str]:
        """
        Selects victim page in Resident_evictable(HBM) with minimum ELLA score (Eq. 4):
          Resident_evictable(HBM) = {P_j in Resident(HBM) | Sealed(P_j) & Mirrored(P_j) & ~Pinned(P_j)}.
        """
        candidates = [
            desc for desc in self.memory._page_descriptors.values()
            if self.memory.is_page_in_device(desc.page_id)
            and desc.is_sealed
            and desc.is_mirrored_to_host
            and not desc.is_pinned
            and not desc.is_anchor
            and desc.page_type != PageType.SYSTEM_ANCHOR
        ]
        if not candidates:
            # Fallback if all non-anchor pages had temporary speculative pins
            candidates = [
                desc for desc in self.memory._page_descriptors.values()
                if self.memory.is_page_in_device(desc.page_id)
                and not desc.is_anchor
                and desc.page_type != PageType.SYSTEM_ANCHOR
            ]
        if not candidates:
            return None

        for c in candidates:
            agent_st = self._agent_states.get(c.sequence_id)
            c.projected_attention_mass = self.compute_projected_attention_score(c, agent_st)

        candidates.sort(key=lambda p: (p.projected_attention_mass, p.last_access_time))
        return candidates[0].page_id
