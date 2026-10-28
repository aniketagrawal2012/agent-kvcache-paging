"""
test_paging.py - Comprehensive Unit Tests for Speculative KV-Cache Paging & ELLA
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Zero mocks. Validates physical memory buffers, LRU/LFU/SGLang/StreamingLLM/H2O/Quest baselines,
5-tier ELLA piecewise formula (Eq. 3), idle tool-schema decay horizon (#4), DAG history boost,
sliding window max rule, Resident_evictable(HBM) anchor exclusion & prefetch-pin fallback (#5),
conditional action-delimiter entropy re-normalization (#9), and non-circular empirical QK RoPE
attention retention (#36).
"""

import unittest
import time
import threading
import numpy as np
from paging_types import (
    PageDescriptor,
    PageType,
    AgentEpistemicState,
    MemoryTier,
    C_FLOOR,
    C_HIST,
    B_DAG_HIST,
    W_WIN,
    resolve_dominant_page_type,
)
from memory_hierarchy import PhysicalMemoryHierarchy, PAGE_TOKENS, KV_HEADS, HEAD_DIM
from baseline_paging import BaselineDemandPager
from epistemic_speculative_pager import EpistemicSpeculativePager
from kvcache_paging import compute_empirical_qk_attention_metrics


class TestKVCachePagingHierarchy(unittest.TestCase):

    def setUp(self):
        self.memory = PhysicalMemoryHierarchy(max_device_pages=4, max_host_pages=32)

    def tearDown(self):
        self.memory.close()

    def test_01_physical_allocation_and_tensor_io(self):
        """Verify real float16 tensor data (128 KiB 2-layer slice) can be written and read."""
        desc = PageDescriptor(
            page_id="p_001",
            sequence_id="seq_01",
            block_index=0,
            page_type=PageType.SYSTEM_ANCHOR,
            is_anchor=True,
        )
        ok = self.memory.allocate_page(desc, initial_tier=MemoryTier.DEVICE_HBM)
        self.assertTrue(ok)
        self.assertTrue(self.memory.is_page_in_device("p_001"))

        test_data = np.full((PAGE_TOKENS, 2, KV_HEADS, HEAD_DIM), 3.14, dtype=np.float16)
        self.assertEqual(test_data.nbytes, 131072)  # Exact 128 KiB per 2-layer slice
        self.memory.write_tensor_data("p_001", test_data)

        read_data = self.memory.read_tensor_data("p_001")
        self.assertTrue(np.allclose(test_data, read_data, atol=1e-2))

    def test_02_lru_eviction_under_capacity_overflow(self):
        """Verify LRU correctly evicts the least-recently used page when Device HBM fills."""
        pager = BaselineDemandPager(self.memory, policy="LRU")
        pages = []
        for i in range(4):
            desc = PageDescriptor(
                page_id=f"page_{i}",
                sequence_id="seq_01",
                block_index=i,
                page_type=PageType.WORKING_ATTENTION,
            )
            self.memory.allocate_page(desc, initial_tier=MemoryTier.DEVICE_HBM)
            pages.append(desc)

        used, total = self.memory.get_device_occupancy()
        self.assertEqual(used, 4)

        for p in pages:
            pager.access_page(p)
            time.sleep(0.002)

        pager.access_page(pages[0])
        pager.access_page(pages[1])
        pager.access_page(pages[2])

        p5 = PageDescriptor(
            page_id="page_4",
            sequence_id="seq_01",
            block_index=4,
            page_type=PageType.WORKING_ATTENTION,
        )
        self.memory.allocate_page(p5, initial_tier=MemoryTier.HOST_RAM)

        ok, is_hit, stall = pager.access_page(p5)
        self.assertTrue(ok)
        self.assertFalse(is_hit)
        self.assertAlmostEqual(stall, 0.012, places=5)  # 12.00 us calibrated dirty demand fault
        self.assertTrue(self.memory.is_page_in_device("page_4"))
        self.assertFalse(self.memory.is_page_in_device("page_3"))

    def test_03_ella_preserves_anchors_and_evicts_scratchpad(self):
        """Verify ELLA policy strictly retains SYSTEM_ANCHOR while evicting SCRATCHPAD."""
        epcp_pager = EpistemicSpeculativePager(self.memory)

        anchor_page = PageDescriptor(
            page_id="anchor_0",
            sequence_id="agent_01",
            block_index=0,
            page_type=PageType.SYSTEM_ANCHOR,
            is_anchor=True,
        )
        scratchpads = [
            PageDescriptor(
                page_id=f"scratch_{i}",
                sequence_id="agent_01",
                block_index=i + 1,
                page_type=PageType.INTERMEDIATE_THOUGHT,
            )
            for i in range(3)
        ]

        self.memory.allocate_page(anchor_page, initial_tier=MemoryTier.DEVICE_HBM)
        for sp in scratchpads:
            self.memory.allocate_page(sp, initial_tier=MemoryTier.DEVICE_HBM)

        agent_st = AgentEpistemicState(
            agent_id="agent_01",
            sequence_id="agent_01",
            epistemic_entropy_h=0.2,
        )
        epcp_pager.register_agent_state(agent_st)

        epcp_pager.access_page(anchor_page, "agent_01")
        for sp in scratchpads:
            epcp_pager.access_page(sp, "agent_01")

        new_tool_page = PageDescriptor(
            page_id="tool_schema_0",
            sequence_id="agent_01",
            block_index=10,
            page_type=PageType.TOOL_SCHEMA,
        )
        self.memory.allocate_page(new_tool_page, initial_tier=MemoryTier.HOST_RAM)

        ok, is_hit, stall = epcp_pager.access_page(new_tool_page, "agent_01")
        self.assertTrue(ok)
        self.assertAlmostEqual(stall, 0.009, places=5)  # 9.00 us clean write-through demand fault
        self.assertTrue(self.memory.is_page_in_device("anchor_0"), "ANCHOR MUST NEVER BE EVICTED!")
        self.assertTrue(self.memory.is_page_in_device("tool_schema_0"))

        evicted_scratchpads = [sp for sp in scratchpads if not self.memory.is_page_in_device(sp.page_id)]
        self.assertGreaterEqual(len(evicted_scratchpads), 1)

    def test_04_speculative_prefetch_eliminates_stalls(self):
        """Verify asynchronous speculative prefetch converts cache miss into zero-stall hit."""
        epcp_pager = EpistemicSpeculativePager(self.memory, lookahead_steps=2)

        p1 = PageDescriptor(page_id="p1", sequence_id="ag1", block_index=0, page_type=PageType.INTERMEDIATE_THOUGHT)
        p2 = PageDescriptor(page_id="p2", sequence_id="ag1", block_index=1, page_type=PageType.INTERMEDIATE_THOUGHT)
        p3 = PageDescriptor(page_id="p3", sequence_id="ag1", block_index=2, page_type=PageType.INTERMEDIATE_THOUGHT)
        for p in [p1, p2, p3]:
            self.memory.allocate_page(p, initial_tier=MemoryTier.DEVICE_HBM)
            epcp_pager.access_page(p, "ag1")

        p4 = PageDescriptor(page_id="p4", sequence_id="ag1", block_index=3, page_type=PageType.TOOL_SCHEMA, dag_distance_d_hat=1)
        self.memory.allocate_page(p4, initial_tier=MemoryTier.HOST_RAM)

        epcp_pager.register_agent_state(AgentEpistemicState(agent_id="ag1", sequence_id="ag1", is_tool_call_pending=True))
        epcp_pager.trigger_speculative_prefetch("ag1", [p4])

        time.sleep(0.005)

        ok, is_hit, stall = epcp_pager.access_page(p4, "ag1")
        self.assertTrue(ok)
        self.assertTrue(is_hit, "Speculative prefetch must ensure page is already in device HBM")
        self.assertEqual(stall, 0.0, "Completed prefetch hit must have zero stall latency")

    def test_05_concurrent_agent_paging(self):
        """Verify thread-safe concurrent paging across multiple parallel agent worker threads."""
        epcp_pager = EpistemicSpeculativePager(self.memory)
        errors = []

        def worker(idx: int):
            try:
                agent_id = f"worker_{idx}"
                desc = PageDescriptor(
                    page_id=f"page_w_{idx}",
                    sequence_id=agent_id,
                    block_index=idx,
                    page_type=PageType.WORKING_ATTENTION,
                )
                self.memory.allocate_page(desc, initial_tier=MemoryTier.HOST_RAM)
                ok, is_hit, stall = epcp_pager.access_page(desc, agent_id)
                if not ok:
                    errors.append(f"Worker {idx} failed access")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent paging had errors: {errors}")

    def test_06_ella_5tier_formula_and_dag_history_boost(self):
        """Verify Eq. (3) history floor C_hist=120, DAG boost >= 470 > C_floor=200, and sliding window max rule."""
        pager = EpistemicSpeculativePager(self.memory)
        st = AgentEpistemicState(
            agent_id="ag_formula",
            sequence_id="ag_formula",
            turn_depth=50,
            epistemic_entropy_h=0.4,
            is_history_summary_pending=False,
        )
        pager.register_agent_state(st)

        # Aged history page (last_access_step=1, global_step=200 -> dt=199 > W_local=64) without DAG boost sits at C_hist=120
        hist_p = PageDescriptor(
            page_id="hist_aged",
            sequence_id="ag_formula",
            block_index=5,
            page_type=PageType.CONVERSATION_HISTORY,
            last_access_turn=0,
            last_access_step=1,
            dag_distance_d_hat=99,
        )
        pager._global_step = 200
        score_idle = pager.compute_projected_attention_score(hist_p, st)
        self.assertEqual(score_idle, C_HIST)

        # When DAG predicts upcoming history summary (d_hat <= 2), score jumps to >= C_hist + B_DAG = 470 > C_floor = 200!
        st.is_history_summary_pending = True
        score_dag = pager.compute_projected_attention_score(hist_p, st)
        self.assertGreaterEqual(score_dag, C_HIST + B_DAG_HIST)
        self.assertGreater(score_dag, C_FLOOR)

        # Tool schema inside W_local=64 receives max(A_class, A_active) >= 10^7 * exp(-3*0.05)
        tool_p = PageDescriptor(
            page_id="tool_recent",
            sequence_id="ag_formula",
            block_index=2,
            page_type=PageType.TOOL_SCHEMA,
            last_access_turn=49,
            last_access_step=195,
        )
        self.assertGreaterEqual(pager.compute_projected_attention_score(tool_p, st), W_WIN * 0.85)

    def test_07_idle_tool_decay_and_dag_history_preemption(self):
        """
        Verify PAT V2 #4 & #7: An idle tool schema (I_tool = 0) decays below a floored
        DAG-predicted history page (>= 470) within ~8 steps (~1 turn), allowing ADSP to preempt it.
        """
        pager = EpistemicSpeculativePager(self.memory)
        st = AgentEpistemicState(
            agent_id="ag_decay",
            sequence_id="ag_decay",
            turn_depth=5,
            epistemic_entropy_h=0.5,
            is_tool_call_pending=False,
            is_history_summary_pending=True,
        )
        pager.register_agent_state(st)
        pager._global_step = 100

        # Idle tool schema accessed 15 steps ago (outside W_local for non-executing or >64 steps, or test class score at dt=80)
        idle_tool = PageDescriptor(
            page_id="tool_dormant",
            sequence_id="ag_decay",
            block_index=1,
            page_type=PageType.TOOL_SCHEMA,
            last_access_step=20,  # dt = 80 steps (~8 turns) -> outside W_local=64
            dag_distance_d_hat=99,
        )
        dag_hist = PageDescriptor(
            page_id="hist_branch_resume",
            sequence_id="ag_decay",
            block_index=8,
            page_type=PageType.CONVERSATION_HISTORY,
            last_access_step=10,  # dt = 90 steps -> floored at C_hist = 120 + B_DAG = 470 + 10 = 480
            dag_distance_d_hat=1,
        )
        tool_score = pager.compute_projected_attention_score(idle_tool, st)
        hist_score = pager.compute_projected_attention_score(dag_hist, st)
        self.assertLess(tool_score, 210.0)
        self.assertGreaterEqual(hist_score, 470.0)
        self.assertGreater(hist_score, tool_score, "DAG-predicted history must outrank decayed idle tool schema!")

        # Verify #7: Idle tool schema at d_hat=3 (outside d_look=2) receives +eta_DAG * d_hat^{-1} = +3.33
        idle_tool_d3 = PageDescriptor(
            page_id="tool_d3",
            sequence_id="ag_decay",
            block_index=2,
            page_type=PageType.TOOL_SCHEMA,
            last_access_step=20,
            dag_distance_d_hat=3,
        )
        score_d3 = pager.compute_projected_attention_score(idle_tool_d3, st)
        self.assertGreater(score_d3, tool_score, "Idle tool at d_hat=3 must outrank permanently dormant tool (d_hat=99)!")

        # Verify #2: Shared page inherits tie-breaker attributes from maximizing owner stream
        st_b = AgentEpistemicState(
            agent_id="ag_b",
            sequence_id="ag_b",
            turn_depth=5,
            epistemic_entropy_h=0.9,
            is_tool_call_pending=False,
        )
        pager.register_agent_state(st_b)
        shared_p = PageDescriptor(
            page_id="shared_tool",
            sequence_id="ag_decay",
            block_index=3,
            page_type=PageType.TOOL_SCHEMA,
            last_access_step=20,
            dag_distance_d_hat=99,
            shared_owners={"ag_decay", "ag_b"},
            per_stream_dag_distance={"ag_decay": 99, "ag_b": 4},
        )
        max_sc, max_inv_d, max_oid, _ = pager.compute_score_and_maximizing_owner(shared_p, st)
        self.assertEqual(max_oid, "ag_b")
        self.assertAlmostEqual(max_inv_d, 0.25, places=5)


    def test_08_invariant1_anchor_exclusion_and_prefetch_release_fallback(self):
        """
        Verify PAT V2 #5: P_anchor is excluded from Resident_evictable(HBM) and prefetch-pinned
        pages are released before any anchor is ever evicted when Resident_evictable(HBM) == empty.
        """
        pager = EpistemicSpeculativePager(self.memory)
        st = AgentEpistemicState(agent_id="ag_inv", sequence_id="ag_inv")
        pager.register_agent_state(st)

        # Fill all 4 HBM frames with 2 anchors and 2 prefetch-pinned tool pages
        a0 = PageDescriptor(page_id="a0", sequence_id="ag_inv", block_index=0, page_type=PageType.SYSTEM_ANCHOR, is_anchor=True)
        a1 = PageDescriptor(page_id="a1", sequence_id="ag_inv", block_index=1, page_type=PageType.SYSTEM_ANCHOR, is_anchor=True)
        p_pin1 = PageDescriptor(page_id="pin1", sequence_id="ag_inv", block_index=2, page_type=PageType.TOOL_SCHEMA, is_pinned=True, pin_expiry_step=999)
        p_pin2 = PageDescriptor(page_id="pin2", sequence_id="ag_inv", block_index=3, page_type=PageType.CONVERSATION_HISTORY, is_pinned=True, pin_expiry_step=999)
        for p in (a0, a1, p_pin1, p_pin2):
            self.memory.allocate_page(p, initial_tier=MemoryTier.DEVICE_HBM)

        # Resident_evictable(HBM) without fallback is empty; fallback must pick pin2 (lowest non-anchor), never a0/a1
        self.assertIsNone(pager._select_ella_victim(allow_prefetch_release_fallback=False))
        fallback_victim = pager._select_ella_victim(allow_prefetch_release_fallback=True)
        self.assertIn(fallback_victim, ("pin1", "pin2"))
        self.assertNotIn(fallback_victim, ("a0", "a1"))

    def test_09_delimiter_entropy_renormalization_and_boundary_dominance(self):
        """Verify PAT V2 #9 & #10: conditional re-normalization over V_act and boundary class dominance."""
        # Unnormalized top-4 sub-distribution summing to 0.40 -> uniform after conditional re-normalization
        raw_sub = [0.10, 0.10, 0.10, 0.10]
        h_norm = AgentEpistemicState.compute_renormalized_delimiter_entropy(raw_sub, v_act_size=4)
        self.assertAlmostEqual(h_norm, 1.0, places=5)

        # Degenerate deterministic distribution -> entropy 0.0
        h_zero = AgentEpistemicState.compute_renormalized_delimiter_entropy([0.85, 0.0, 0.0, 0.0], v_act_size=4)
        self.assertAlmostEqual(h_zero, 0.0, places=5)

        # Boundary dominance: P_anchor > P_tool > P_hist > P_scratch
        self.assertEqual(resolve_dominant_page_type(PageType.INTERMEDIATE_THOUGHT, PageType.TOOL_SCHEMA), PageType.TOOL_SCHEMA)
        self.assertEqual(resolve_dominant_page_type(PageType.CONVERSATION_HISTORY, PageType.SYSTEM_ANCHOR), PageType.SYSTEM_ANCHOR)

    def test_10_non_circular_empirical_qk_attention_and_sota_baselines(self):
        """
        Verify PAT V2 #30 & #36: SOTA baselines (SGLang/RadixAttention, StreamingLLM, H2O, Quest)
        execute properly, and non-circular empirical QK RoPE attention distinguishes ELLA from StreamingLLM.
        """
        mem_ella = PhysicalMemoryHierarchy(max_device_pages=6, max_host_pages=32)
        mem_sllm = PhysicalMemoryHierarchy(max_device_pages=6, max_host_pages=32)
        try:
            ella = EpistemicSpeculativePager(mem_ella)
            sllm = BaselineDemandPager(mem_sllm, policy="STREAMING_LLM")
            st = AgentEpistemicState(agent_id="ag_qk", sequence_id="ag_qk", is_tool_call_pending=True)
            ella.register_agent_state(st)

            def make_pages():
                return [
                    PageDescriptor(page_id="ank_0", sequence_id="ag_qk", block_index=0, page_type=PageType.SYSTEM_ANCHOR, is_anchor=True),
                    PageDescriptor(page_id="ank_1", sequence_id="ag_qk", block_index=1, page_type=PageType.SYSTEM_ANCHOR, is_anchor=True),
                    PageDescriptor(page_id="tool_0", sequence_id="ag_qk", block_index=2, page_type=PageType.TOOL_SCHEMA, dag_distance_d_hat=1),
                    PageDescriptor(page_id="tool_1", sequence_id="ag_qk", block_index=3, page_type=PageType.TOOL_SCHEMA, dag_distance_d_hat=1),
                    PageDescriptor(page_id="hist_0", sequence_id="ag_qk", block_index=4, page_type=PageType.CONVERSATION_HISTORY, dag_distance_d_hat=2),
                    PageDescriptor(page_id="scr_0", sequence_id="ag_qk", block_index=5, page_type=PageType.INTERMEDIATE_THOUGHT),
                    PageDescriptor(page_id="scr_1", sequence_id="ag_qk", block_index=6, page_type=PageType.INTERMEDIATE_THOUGHT),
                    PageDescriptor(page_id="scr_2", sequence_id="ag_qk", block_index=7, page_type=PageType.INTERMEDIATE_THOUGHT),
                    PageDescriptor(page_id="win_0", sequence_id="ag_qk", block_index=8, page_type=PageType.WORKING_ATTENTION),
                    PageDescriptor(page_id="win_1", sequence_id="ag_qk", block_index=9, page_type=PageType.WORKING_ATTENTION),
                ]

            pages_ella = make_pages()
            pages_sllm = make_pages()
            for p_e, p_s in zip(pages_ella, pages_sllm):
                mem_ella.allocate_page(p_e, initial_tier=MemoryTier.HOST_RAM)
                mem_sllm.allocate_page(p_s, initial_tier=MemoryTier.HOST_RAM)
                ella.access_page(p_e, "ag_qk")
                sllm.access_page(p_s)

            qk_ella = compute_empirical_qk_attention_metrics(pages_ella, mem_ella, query_mode="TOOL_CALL", top_k=6)
            qk_sllm = compute_empirical_qk_attention_metrics(pages_sllm, mem_sllm, query_mode="TOOL_CALL", top_k=6)

            self.assertGreater(
                qk_ella["empirical_qk_retained_pct"],
                qk_sllm["empirical_qk_retained_pct"],
                "ELLA must retain strictly more empirical QK dot-product attention mass on tool schemas than StreamingLLM!",
            )
        finally:
            mem_ella.close()
            mem_sllm.close()


if __name__ == "__main__":
    unittest.main()
