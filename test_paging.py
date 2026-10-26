"""
test_paging.py - Comprehensive Unit Tests for Speculative KV-Cache Paging & ELLA
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Zero mocks. Validates physical memory buffers, LRU/LFU baselines,
5-tier ELLA piecewise formula (Eq. 3), DAG history boost, sliding window max rule,
Resident_evictable(HBM) lock bits, and Epistemic Speculative Paging (ELLA) with prefetching.
"""

import unittest
import time
import threading
import numpy as np
from paging_types import PageDescriptor, PageType, AgentEpistemicState, MemoryTier, C_FLOOR, W_WIN
from memory_hierarchy import PhysicalMemoryHierarchy, PAGE_TOKENS, KV_HEADS, HEAD_DIM
from baseline_paging import BaselineDemandPager
from epistemic_speculative_pager import EpistemicSpeculativePager


class TestKVCachePagingHierarchy(unittest.TestCase):

    def setUp(self):
        self.memory = PhysicalMemoryHierarchy(max_device_pages=4, max_host_pages=32)

    def tearDown(self):
        self.memory.close()

    def test_01_physical_allocation_and_tensor_io(self):
        """Verify real float16 tensor data can be written and read from physical buffers."""
        desc = PageDescriptor(
            page_id="p_001",
            sequence_id="seq_01",
            block_index=0,
            page_type=PageType.SYSTEM_ANCHOR,
            is_anchor=True
        )
        ok = self.memory.allocate_page(desc, initial_tier=MemoryTier.DEVICE_HBM)
        self.assertTrue(ok)
        self.assertTrue(self.memory.is_page_in_device("p_001"))

        test_data = np.full((PAGE_TOKENS, 2, KV_HEADS, HEAD_DIM), 3.14, dtype=np.float16)
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
                page_type=PageType.WORKING_ATTENTION
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
            page_type=PageType.WORKING_ATTENTION
        )
        self.memory.allocate_page(p5, initial_tier=MemoryTier.HOST_RAM)

        ok, is_hit, stall = pager.access_page(p5)
        self.assertTrue(ok)
        self.assertFalse(is_hit)
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
            is_anchor=True
        )
        scratchpads = [
            PageDescriptor(
                page_id=f"scratch_{i}",
                sequence_id="agent_01",
                block_index=i + 1,
                page_type=PageType.INTERMEDIATE_THOUGHT
            )
            for i in range(3)
        ]

        self.memory.allocate_page(anchor_page, initial_tier=MemoryTier.DEVICE_HBM)
        for sp in scratchpads:
            self.memory.allocate_page(sp, initial_tier=MemoryTier.DEVICE_HBM)

        agent_st = AgentEpistemicState(
            agent_id="agent_01",
            sequence_id="agent_01",
            epistemic_entropy_h=0.2
        )
        epcp_pager.register_agent_state(agent_st)

        epcp_pager.access_page(anchor_page, "agent_01")
        for sp in scratchpads:
            epcp_pager.access_page(sp, "agent_01")

        new_tool_page = PageDescriptor(
            page_id="tool_schema_0",
            sequence_id="agent_01",
            block_index=10,
            page_type=PageType.TOOL_SCHEMA
        )
        self.memory.allocate_page(new_tool_page, initial_tier=MemoryTier.HOST_RAM)

        ok, is_hit, stall = epcp_pager.access_page(new_tool_page, "agent_01")
        self.assertTrue(ok)
        self.assertTrue(self.memory.is_page_in_device("anchor_0"), "ANCHOR MUST NEVER BE EVICTED!")
        self.assertTrue(self.memory.is_page_in_device("tool_schema_0"))

        evicted_scratchpads = [sp for sp in scratchpads if not self.memory.is_page_in_device(sp.page_id)]
        self.assertGreaterEqual(len(evicted_scratchpads), 1)

    def test_04_speculative_prefetch_eliminates_stalls(self):
        """Verify asynchronous speculative prefetch converts cache miss into zero-stall hit."""
        epcp_pager = EpistemicSpeculativePager(self.memory, lookahead_steps=2)

        p1 = PageDescriptor(page_id="p1", sequence_id="ag1", block_index=0, page_type=PageType.WORKING_ATTENTION)
        p2 = PageDescriptor(page_id="p2", sequence_id="ag1", block_index=1, page_type=PageType.WORKING_ATTENTION)
        p3 = PageDescriptor(page_id="p3", sequence_id="ag1", block_index=2, page_type=PageType.WORKING_ATTENTION)
        for p in [p1, p2, p3]:
            self.memory.allocate_page(p, initial_tier=MemoryTier.DEVICE_HBM)
            epcp_pager.access_page(p, "ag1")

        p4 = PageDescriptor(page_id="p4", sequence_id="ag1", block_index=3, page_type=PageType.TOOL_SCHEMA)
        self.memory.allocate_page(p4, initial_tier=MemoryTier.HOST_RAM)

        epcp_pager.register_agent_state(AgentEpistemicState(agent_id="ag1", sequence_id="ag1", is_tool_call_pending=True))
        epcp_pager.trigger_speculative_prefetch("ag1", [p4])

        time.sleep(0.005)

        ok, is_hit, stall = epcp_pager.access_page(p4, "ag1")
        self.assertTrue(ok)
        self.assertTrue(is_hit, "Speculative prefetch must ensure page is already in device HBM")
        self.assertLess(stall, 1.0, "Stall latency must be virtually zero")

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
                    block_index=0,
                    page_type=PageType.WORKING_ATTENTION
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
        """Verify Eq. (3) history DAG boost > C_floor and sliding window max(A_class, A_active) rule."""
        pager = EpistemicSpeculativePager(self.memory)
        st = AgentEpistemicState(
            agent_id="ag_formula",
            sequence_id="ag_formula",
            turn_depth=50,
            epistemic_entropy_h=0.4,
            is_history_summary_pending=False,
        )
        pager.register_agent_state(st)

        # Aged history page (last_access_turn=0, turn_depth=50) without DAG boost sits at C_floor=200
        hist_p = PageDescriptor(
            page_id="hist_aged",
            sequence_id="ag_formula",
            block_index=5,
            page_type=PageType.CONVERSATION_HISTORY,
            last_access_turn=0,
            last_access_step=0,
        )
        pager._global_step = 200
        score_idle = pager.compute_projected_attention_score(hist_p, st)
        self.assertEqual(score_idle, C_FLOOR)

        # When DAG predicts upcoming history summary (d_hat <= 2), score jumps above C_floor!
        st.is_history_summary_pending = True
        score_dag = pager.compute_projected_attention_score(hist_p, st)
        self.assertGreater(score_dag, C_FLOOR + 300.0)

        # Tool schema inside W_local=64 receives max(A_class, W_win=1000)
        tool_p = PageDescriptor(
            page_id="tool_recent",
            sequence_id="ag_formula",
            block_index=2,
            page_type=PageType.TOOL_SCHEMA,
            last_access_turn=49,
            last_access_step=190,
        )
        self.assertGreaterEqual(pager.compute_projected_attention_score(tool_p, st), W_WIN)


if __name__ == "__main__":
    unittest.main()
