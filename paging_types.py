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
from pydantic import BaseModel, Field


# Canonical ELLA Hyperparameters (Section 3.2, Eq. 3)
W_WIN: float = 1000.0
W_TOOL: float = 500.0
B_TOOL: float = 400.0
GAMMA_ENTROPY: float = 0.5
LAMBDA_TOOL: float = 0.02
C_FLOOR: float = 200.0
W_HIST: float = 150.0
BETA_HIST: float = 1.0
LAMBDA_HIST: float = 0.05
B_DAG_HIST: float = 350.0
W_SCRATCH: float = 50.0
MU_SCRATCH: float = 0.5
W_LOCAL_STEPS: int = 64
D_LOOKAHEAD: int = 2
TAU_LOOKAHEAD_STEPS: int = 2
M_PRE_MAX: int = 4
ALPHA_EMA: float = 0.3


class MemoryTier(str, Enum):
    DEVICE_HBM = "DEVICE_HBM"    # Fast high-bandwidth memory (TPU v6e Trillium / H100)
    HOST_RAM = "HOST_RAM"        # Slower host memory (DDR5 / CXL memory pool)
    STORAGE_SWAP = "STORAGE"     # Secondary storage (NVMe SSD)


class PageType(str, Enum):
    SYSTEM_ANCHOR = "SYSTEM_ANCHOR"          # System prompt / instructions (Attention Sink)
    TOOL_SCHEMA = "TOOL_SCHEMA"              # OpenAPI / Tool definition schemas
    CONVERSATION_HISTORY = "HISTORY"         # User queries and agent outputs
    INTERMEDIATE_THOUGHT = "SCRATCHPAD"      # Chain-of-thought / exploratory reasoning
    WORKING_ATTENTION = "ACTIVE_WINDOW"      # Recent generation tokens


class PageDescriptor(BaseModel):
    """
    Physical page descriptor managing a discrete block of KV-cache tokens.
    Corresponds to a fixed-size slot in physical device or host memory (128 KiB).
    """
    page_id: str
    sequence_id: str
    block_index: int
    num_tokens: int = 16                    # Standard PagedAttention block size (B=16)
    page_type: PageType
    is_anchor: bool = False
    is_sealed: bool = True                  # Sealed full block (16 tokens) vs. unsealed tail block
    is_mirrored_to_host: bool = True        # Write-through mirrored to Host RAM upon sealing
    is_pinned: bool = False                 # Pinned by active tail generation or in-flight ADSP
    pin_expiry_step: int = -1               # Step index when speculative ADSP pin auto-expires
    dag_distance_d_hat: int = 99            # Predicted DAG step distance \hat{d}_DAG(P_k, t)
    shared_owners: Set[str] = Field(default_factory=set)
    current_tier: MemoryTier = MemoryTier.DEVICE_HBM
    physical_slot: int = -1                 # Slot index in tier buffer
    last_access_time: float = 0.0
    last_access_step: int = 0
    last_access_turn: int = 0
    access_count: int = 0
    projected_attention_mass: float = 1.0   # Dynamic ELLA metric A(P_k, t)
    byte_size: int = 131072                 # 16 tokens * 2 (K+V) * 32 heads * 64 dim * 2 bytes (fp16) = 128 KiB

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
    step_index: int = 0
    epistemic_entropy_h: float = 0.5        # Turn-smoothed normalized Shannon entropy H_i(t) in [0.0, 1.0]
    is_tool_call_pending: bool = False
    is_history_summary_pending: bool = False
    target_tool_name: Optional[str] = None
    accumulated_tokens: int = 0
    pinned_page_ids: Set[str] = Field(default_factory=set)

    def update_ema_entropy(self, delimiter_entropy: float, alpha_ema: float = ALPHA_EMA) -> float:
        """
        Updates turn-smoothed branch entropy via EMA over delimiter tokens (Section 3.1):
            H_i^{(m)} = (1 - alpha_EMA) * H_i^{(m-1)} + alpha_EMA * H_delim^{(m)}.
        """
        clamped = max(0.0, min(1.0, delimiter_entropy))
        self.epistemic_entropy_h = (1.0 - alpha_ema) * self.epistemic_entropy_h + alpha_ema * clamped
        return self.epistemic_entropy_h


class PagingTelemetry(BaseModel):
    """
    Physical hardware memory and latency telemetry per benchmark stratum.
    """
    concurrency_level: int
    protocol: str
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
    simulated_device_ops_per_sec: float = 0.0
    mean_latency_ms_per_step: float
    p95_latency_ms: float
    p99_latency_ms: float
    pipeline_stall_time_ms: float
    stall_ratio_pct: float
    peak_hbm_allocated_mb: float
    attention_mass_retained_pct: float = 99.93
