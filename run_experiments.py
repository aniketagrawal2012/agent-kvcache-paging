"""
run_experiments.py - Main Entrypoint for SE-Paging & ELLA Experimental Evaluation
Docket: IDF-GGL-2026-KVCACHE-SPEC-PAGING
Author: Aniket Agrawal (aniketagrawal@google.com)
Assignee: Google LLC

Executes the full 1:32 physical tensor-arena paging benchmark suite, SOTA baseline
comparisons, non-circular empirical QK RoPE attention evaluation, hyperparameter
sensitivity grid, and live Vertex AI (`gemini-2.5-flash`) trace replay.
"""

from run_live_benchmark import (
    export_canonical_telemetry_csv,
    render_publication_plots,
    run_live_gemini_kvcache_trace_replay,
)

if __name__ == "__main__":
    export_canonical_telemetry_csv()
    render_publication_plots()
    run_live_gemini_kvcache_trace_replay(concurrency=20, turns_per_agent=6)
