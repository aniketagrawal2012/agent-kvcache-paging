"""
baseline_paging.py - Classical Demand Paging with LRU / LFU Eviction
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Implements classical reactive Demand Paging (analogous to standard vLLM / TensorRT-LLM host swap).
Suffers pipeline stalls whenever a page fault occurs during token generation.
"""

from __future__ import annotations
import time
import threading
from typing import Dict, List, Optional, Tuple
from paging_types import PageDescriptor, MemoryTier
from memory_hierarchy import PhysicalMemoryHierarchy


class BaselineDemandPager:
    """
    Standard demand pager with configurable eviction policy (LRU or LFU).
    """

    def __init__(self, memory: PhysicalMemoryHierarchy, policy: str = "LRU"):
        self.memory = memory
        self.policy = policy.upper()
        self._lock = threading.Lock()
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
        t_start = time.perf_counter()
        page_id = page_desc.page_id

        with self._lock:
            self.total_accesses += 1
            now = time.time()
            page_desc.last_access_time = now
            page_desc.access_count += 1

            if self.memory.is_page_in_device(page_id):
                self.hits += 1
                return True, True, 0.0

            # Cache Miss: Synchronous Demand Paging required
            self.misses += 1

            # Check if device HBM has space
            used, total = self.memory.get_device_occupancy()
            if used >= total:
                # Evict a victim page
                victim_id = self._select_victim()
                if victim_id:
                    self.memory.evict_device_page_to_host(victim_id)
                    self.evictions += 1

            # Demand swap requested page into Device HBM
            ok = self.memory.demand_swap_to_device(page_id)
            stall_ms = (time.perf_counter() - t_start) * 1000.0
            self.total_stall_ms += stall_ms
            return ok, False, stall_ms

    def _select_victim(self) -> Optional[str]:
        """Selects a victim page currently residing in Device HBM."""
        candidates = [
            desc for desc in self.memory._page_descriptors.values()
            if self.memory.is_page_in_device(desc.page_id)
        ]
        if not candidates:
            return None

        if self.policy == "LFU":
            candidates.sort(key=lambda p: (p.access_count, p.last_access_time))
        else:  # Default: LRU
            candidates.sort(key=lambda p: p.last_access_time)

        return candidates[0].page_id
