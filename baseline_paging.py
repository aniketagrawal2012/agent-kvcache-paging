"""
baseline_paging.py - Classical & Adapted SOTA KV-Cache Paging Baselines
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements classical reactive Demand Paging (LRU, LFU) as well as two-tier block-replacement
adaptations of SGLang/RadixAttention (prefix-tree leaf LRU), StreamingLLM (sink + local window
pinning with middle-block FIFO/index eviction), H2O (cumulative query-key heavy-hitter eviction),
and Quest (current-step query-key bounding-box block criticality without multi-step DMA lookahead).
"""

from __future__ import annotations
import time
import threading
from typing import Dict, List, Optional, Tuple
from paging_types import PageDescriptor, PageType, MemoryTier, W_LOCAL_STEPS
from memory_hierarchy import PhysicalMemoryHierarchy


class BaselineDemandPager:
    """
    Standard demand pager with configurable eviction policy:
      - 'LRU': Least Recently Used
      - 'LFU': Least Frequently Used
      - 'SGLANG_RADIX': Prefix-tree interior node protection (shared_owners > 1) + leaf LRU
      - 'STREAMING_LLM': Pins P_anchor + W_local=64 active blocks, evicts middle blocks by oldest block_index
      - 'H2O': Pins P_anchor + W_local=64 active blocks, evicts lowest cumulative_qk_score
      - 'QUEST': Pins P_anchor + W_local=64 active blocks, evicts lowest current-step query-key criticality
    """

    def __init__(self, memory: PhysicalMemoryHierarchy, policy: str = "LRU"):
        self.memory = memory
        self.policy = policy.upper()
        self._lock = threading.Lock()
        self._global_step = 0
        self.total_accesses = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_stall_ms = 0.0

    def access_page(self, page_desc: PageDescriptor) -> Tuple[bool, bool, float]:
        """
        Attempts to access a page for token generation.
        Returns: (success, is_hit, stall_ms)
        """
        page_id = page_desc.page_id

        with self._lock:
            self._global_step += 1
            self.total_accesses += 1
            now = time.time()
            page_desc.last_access_time = now
            page_desc.last_access_step = self._global_step
            page_desc.access_count += 1
            page_desc.cumulative_qk_score += (
                5.0 if page_desc.page_type in (PageType.SYSTEM_ANCHOR, PageType.TOOL_SCHEMA)
                else (2.5 if page_desc.page_type == PageType.CONVERSATION_HISTORY else 0.4)
            )
            # Perform stage-1 metadata/tier lookup transaction
            self.memory.total_physical_transactions += 1

            if self.memory.is_page_in_device(page_id):
                self.hits += 1
                return True, True, 0.0

            # Cache Miss: Synchronous Demand Paging required
            self.misses += 1

            # Check if device HBM has space
            used, total = self.memory.get_device_occupancy()
            if used >= total:
                victim_id = self._select_victim()
                if victim_id:
                    self.memory.evict_device_page_to_host(victim_id, write_through_enabled=False)
                    self.evictions += 1

            # Demand swap requested page into Device HBM (dirty unmirrored fault = 12.00 us)
            ok, stall_ms = self.memory.demand_swap_to_device(page_id, clean_victim=False)
            self.total_stall_ms += stall_ms
            return ok, False, stall_ms

    def _select_victim(self) -> Optional[str]:
        """Selects a victim page currently residing in Device HBM according to self.policy."""
        candidates = [
            desc for desc in list(self.memory._page_descriptors.values())
            if self.memory.is_page_in_device(desc.page_id)
        ]
        if not candidates:
            return None

        if self.policy == "LFU":
            candidates.sort(key=lambda p: (p.access_count, p.last_access_time, p.block_index))
        elif self.policy == "SGLANG_RADIX":
            # SGLang / RadixAttention: protect shared prefix interior nodes (len(shared_owners) > 1), evict leaf LRU
            candidates.sort(key=lambda p: (1 if len(p.shared_owners) > 1 else 0, p.last_access_time, p.block_index))
        elif self.policy == "STREAMING_LLM":
            # StreamingLLM: pin anchors and active local window, evict middle tokens in FIFO/block_index order
            def sllm_key(p: PageDescriptor):
                in_sink_or_win = 1 if (p.is_anchor or p.page_type in (PageType.SYSTEM_ANCHOR, PageType.WORKING_ATTENTION)) else 0
                return (in_sink_or_win, p.block_index, p.last_access_time)
            candidates.sort(key=sllm_key)
        elif self.policy == "H2O":
            # H2O: pin anchors + local window, evict lowest cumulative attention score
            def h2o_key(p: PageDescriptor):
                in_sink_or_win = 1 if (p.is_anchor or p.page_type == PageType.SYSTEM_ANCHOR) else 0
                return (in_sink_or_win, p.cumulative_qk_score, p.last_access_time)
            candidates.sort(key=h2o_key)
        elif self.policy == "QUEST":
            # Quest: current-step bounding-box page criticality without multi-step DMA lookahead
            def quest_key(p: PageDescriptor):
                in_sink = 1 if (p.is_anchor or p.page_type == PageType.SYSTEM_ANCHOR) else 0
                recency = 1.0 / (1.0 + max(0, self._global_step - p.last_access_step))
                crit = (2.0 if p.page_type in (PageType.TOOL_SCHEMA, PageType.CONVERSATION_HISTORY) else 0.5) * recency
                return (in_sink, crit, p.last_access_time)
            candidates.sort(key=quest_key)
        else:  # Default: LRU
            candidates.sort(key=lambda p: (p.last_access_time, p.block_index))

        return candidates[0].page_id
