"""
paging_types.py - Pydantic Data Contracts for Speculative KV-Cache Paging & ELLA
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Defines strict types for memory tiers, page descriptors, agent epistemic states,
and hardware memory telemetry. Zero mocks.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Set
import numpy as np
from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    DEVICE_HBM = "DEVICE_HBM"    # Fast high-bandwidth memory (TPU v6e Trillium / H100)
    HOST_RAM = "HOST_RAM"        # Slower host memory (DDR5 / CXL memory pool)
    STORAGE_SWAP = "STORAGE"    # Secondary storage (NVMe SSD)


class PageType(str, Enum):
    SYSTEM_ANCHOR = "SYSTEM_ANCHOR"          # System prompt / instructions (Attention Sink)
    TOOL_SCHEMA = "TOOL_SCHEMA"              # OpenAPI / Tool definition schemas
    CONVERSATION_HISTORY = "HISTORY"         # User queries and agent outputs
    INTERMEDIATE_THOUGHT = "SCRATCHPAD"      # Chain-of-thought / exploratory reasoning
    WORKING_ATTENTION = "ACTIVE_WINDOW"      # Recent generation tokens


class PageDescriptor(BaseModel):
    """
    Physical page descriptor managing a discrete block of KV-cache tokens.
    Corresponds to a fixed-size slot in physical device or host memory.
    """
    page_id: str
    sequence_id: str
    block_index: int
    num_tokens: int = 16                    # Standard PagedAttention block size (B=16)
    page_type: PageType
    is_anchor: bool = False
    current_tier: MemoryTier = MemoryTier.DEVICE_HBM
    physical_slot: int = -1                 # Slot index in tier buffer
    last_access_time: float = 0.0
    access_count: int = 0
    projected_attention_mass: float = 1.0   # Dynamic ELLA metric
    byte_size: int = 65536                  # 16 tokens * 2 (K+V) * 32 heads * 64 dim * 2 bytes (fp16) = 64KB

    class Config:
        arbitrary_types_allowed = True


class AgentEpistemicState(BaseModel):
    """
    Dynamic semantic state of an active autonomous agent stream.
    Used by the Epistemic Working Set Estimator (EWSE) to project future KV-cache access.
    """
    agent_id: str
    sequence_id: str
    turn_depth: int = 0
    epistemic_entropy_h: float = 0.5        # Shannon entropy of next-token distribution [0.0, 1.0]
    is_tool_call_pending: bool = False
    target_tool_name: Optional[str] = None
    accumulated_tokens: int = 0
    pinned_page_ids: Set[str] = Field(default_factory=set)


class PagingTelemetry(BaseModel):
    """
    Physical hardware memory and latency telemetry per benchmark stratum.
    """
    concurrency_level: int
    protocol: str                           # 'DEMAND_LRU', 'DEMAND_LFU', 'SPECULATIVE_ELLA'
    total_tokens_generated: int
    total_page_accesses: int
    cache_hits: int
    cache_misses: int
    hit_rate_pct: float
    speculative_prefetch_hits: int
    eviction_count: int
    total_dma_bytes_transferred: int
    total_wall_time_sec: float
    throughput_tokens_per_sec: float
    mean_latency_ms_per_step: float
    p95_latency_ms: float
    p99_latency_ms: float
    pipeline_stall_time_ms: float
    stall_ratio_pct: float
    peak_hbm_allocated_mb: float
