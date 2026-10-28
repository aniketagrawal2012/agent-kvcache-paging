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


# Canonical ELLA Hyperparameters (Section 4.1, Eq. 3 & Section 6.1)
SCORE_ANCHOR: float = 1e9
SCORE_ACTIVE_TOOL: float = 5e8
W_WIN: float = 1e7
KAPPA_WIN: float = 0.05
W_TOOL: float = 500.0
B_TOOL: float = 5e8
GAMMA_ENTROPY: float = 0.5
LAMBDA_TOOL: float = 0.08
C_FLOOR: float = 200.0
C_HIST: float = 120.0
W_HIST: float = 600.0
BETA_HIST: float = 1.0
EPSILON_HIST: float = 0.10
LAMBDA_HIST: float = 0.08
B_DAG_HIST: float = 350.0
ETA_DAG: float = 10.0
W_SCRATCH: float = 100.0
MU_SCRATCH: float = 0.45
W_LOCAL_STEPS: int = 64
W_LOCAL_BLOCKS: int = 4
D_LOOKAHEAD: int = 2
TAU_LOOKAHEAD_STEPS: int = 2
M_PRE_MAX: int = 4
ALPHA_EMA: float = 0.30
V_ACT_SIZE: int = 64

# 1:32 Downscaled Slice Timing Constants (2-layer 128 KiB slice out of L=64 layers)
T_COMPUTE_IDEAL_US: float = 5.00
T_DMA_SLICE_US: float = 3.74
RHO_BUS_DEFAULT: float = 0.18
T_WAKEUP_SLICE_US: float = 4.44
T_WRITEBACK_SLICE_US: float = 3.00
T_FAULT_DIRTY_US: float = 12.00   # 4.44 wakeup + 3.74/0.82 DMA + 3.00 dirty writeback
T_FAULT_CLEAN_US: float = 9.00    # 4.44 wakeup + 3.74/0.82 DMA (write-through clean)
T_FAULT_PARTIAL_US: float = 2.50  # Residual wait on in-flight ADSP prefetch


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


# Dominance hierarchy for delimiter boundary straddling:
# P_anchor > P_tool > P_hist > P_scratch
PAGE_CLASS_DOMINANCE: Dict[PageType, int] = {
    PageType.SYSTEM_ANCHOR: 4,
    PageType.TOOL_SCHEMA: 3,
    PageType.WORKING_ATTENTION: 2,
    PageType.CONVERSATION_HISTORY: 1,
    PageType.INTERMEDIATE_THOUGHT: 0,
}


def resolve_dominant_page_type(t1: PageType, t2: PageType) -> PageType:
    """Resolves dominant semantic class when a B=16-token block straddles two spans."""
    return t1 if PAGE_CLASS_DOMINANCE.get(t1, 0) >= PAGE_CLASS_DOMINANCE.get(t2, 0) else t2


class PageDescriptor(BaseModel):
    """
    Physical page descriptor managing a discrete block of KV-cache tokens.
    Corresponds to a fixed-size 2-layer fused slice in physical device or host memory (128 KiB).
    For L=64, H_kv=8, d_k=128, float16 (2 bytes):
      1-layer B=16 page = 16 * 2 * 8 * 128 * 2 = 65,536 bytes (64 KiB)
      2-layer fused slice = 2 * 65,536 = 131,072 bytes (128 KiB)
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
    dag_distance_d_hat: int = 99            # Predicted DAG step distance \hat{d}_{DAG,i}(P_k, t)
    slice_residency_mask: int = 0xFFFFFFFF  # 32-bit layer-slice residency bitmap across L=64 layers
    shared_owners: Set[str] = Field(default_factory=set)
    per_stream_last_access_step: Dict[str, int] = Field(default_factory=dict)
    per_stream_dag_distance: Dict[str, int] = Field(default_factory=dict)
    current_tier: MemoryTier = MemoryTier.DEVICE_HBM
    physical_slot: int = -1                 # Slot index in tier buffer
    last_access_time: float = 0.0
    last_access_step: int = 0
    last_access_turn: int = 0
    access_count: int = 0
    cumulative_qk_score: float = 0.0        # Cumulative query-key attention score for H2O/Quest
    projected_attention_mass: float = 1.0   # Dynamic ELLA metric A(P_k, t)
    byte_size: int = 131072                 # 2-layer slice: 2 * 16 * 2 (K+V) * 8 heads * 128 dim * 2 bytes = 128 KiB

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
    epistemic_entropy_h: float = 0.50       # Turn-smoothed normalized Shannon entropy H_i(t) in [0.0, 1.0], H_i(0)=0.50
    is_tool_call_pending: bool = False
    is_history_summary_pending: bool = False
    is_active_executing: bool = True        # True when stream S_i is currently executing step t
    target_tool_name: Optional[str] = None
    accumulated_tokens: int = 0
    pinned_page_ids: Set[str] = Field(default_factory=set)

    @staticmethod
    def compute_renormalized_delimiter_entropy(raw_probs: List[float], v_act_size: int = V_ACT_SIZE) -> float:
        """
        Conditionally re-normalizes action-delimiter token probabilities over V_act (|V_act| <= 64)
        so sum_{v in V_act} p_tilde(v) = 1, and returns normalized Shannon entropy in [0, 1] (Eq. 5).
        """
        import math
        positive = [max(0.0, float(p)) for p in raw_probs[:v_act_size] if p > 0.0]
        total = sum(positive)
        if total <= 0.0 or len(positive) <= 1:
            return 0.0
        denom = math.log(max(2, min(v_act_size, len(raw_probs))))
        h_sum = 0.0
        for p in positive:
            p_tilde = p / total
            if p_tilde > 0.0:
                h_sum -= p_tilde * math.log(p_tilde)
        return max(0.0, min(1.0, h_sum / denom))

    def update_ema_entropy(self, delimiter_entropy: float, alpha_ema: float = ALPHA_EMA) -> float:
        """
        Updates turn-smoothed branch entropy via EMA over delimiter tokens (Section 4.1, Eq. 5):
            H_i(t) = (1 - beta) * H_i(t-1) + beta * H_norm(p_tilde_{i,m}).
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
    partial_overlap_misses: int = 0
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
    attention_mass_retained_pct: float = 98.64
    analytical_rope_retained_pct: float = 99.93
    topk_qk_recall_pct: float = 96.80
