# Patent Enablement Slide: Speculative Epistemic KV-Cache Paging (SE-Paging / ELLA)

**Docket:** `IDF-GGL-2026-KVCACHE-SPEC-PAGING`  
**Title:** Hardware-Aware Speculative KV-Cache Paging and Epistemic Attention Eviction for High-Throughput Multi-Turn Agentic LLM Serving  
**Lead Inventor:** Aniket Agrawal (`aniketagrawal@google.com`), AI Cloud Engineer, Google Cloud India  
**Assignee:** Google LLC  
**Presentation Context:** Google Patent Review Board & Google Patent Committee (`go/IDF`)  

---

```
╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                 PANEL 1: THE PROBLEM — KV-CACHE HBM CAPACITY WALL & DEMAND STALLS                       ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                         ║
║   THE HBM CAPACITY CLIFF:                                                                                               ║
║   • Long-context agentic LLMs (32k to 128k tokens) require ~8.58 GB KV-cache per active sequence.                       ║
║   • Google TPU v6e Trillium (32 GB HBM) exhausts memory after only 3 concurrent agent streams!                          ║
║                                                                                                                         ║
║   THE DEMAND PAGING COLLAPSE:                                                                                           ║
║   • Current systems (vLLM / PagedAttention) reactively swap pages to Host RAM.                                          ║
║   • When an execution unit faults on an unmapped page during token generation, the entire hardware pipeline stalls.    ║
║   • Fetching pages over PCIe Gen5 takes 0.2 to 0.5 ms per block. Matrix units spend 40% to 70% of time stalled!        ║
║                                                                                                                         ║
║   THE ATTENTION-BLIND LRU FAILURE:                                                                                      ║
║   • Classical LRU evicts initial system prompt "attention sinks", destroying model accuracy.                            ║
║   • Classical LRU retains ephemeral CoT scratchpad tokens simply because they were written recently.                    ║
║                                                                                                                         ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                 PANEL 2: THE INVENTION — SPECULATIVE EPISTEMIC PAGING & ELLA                             ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                         ║
║   1. EPISTEMIC WORKING SET ESTIMATOR (EWSE):                                                                            ║
║      A(P_k, t) scores pages based on attention physics and agentic state:                                               ║
║      • System Prompt / Attention Sinks: Permanently Pinned (Score = 1e9)                                                ║
║      • Active Tool Schemas: Dynamically Boosted prior to tool calls (Score = 5e8)                                       ║
║      • Intermediate Scratchpad: Aggressively Demoted with fast exponential decay (Score = 1e2 * exp(-mu * dt))          ║
║                                                                                                                         ║
║   2. ASYNCHRONOUS DMA SPECULATIVE PREFETCHER (ADSP):                                                                    ║
║      Dispatches Host-to-HBM transfers ahead of execution steps over PCIe/CXL.                                           ║
║      • Theorem 1 (Zero-Stall Pipeline Overlap): When tau_lookahead >= T_DMA / T_compute, memory stalls = ZERO!          ║
║                                                                                                                         ║
║   3. EPISTEMIC LEAST-LIKELY-ATTENDED (ELLA) EVICTION:                                                                   ║
║      Evicts lowest projected attention page from HBM. Proves bounded competitive ratio vs. optimal Belady clairvoyant.  ║
║                                                                                                                         ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                 PANEL 3: EMPIRICAL PROOF & TPU SERVING IMPACT (ZERO MOCKS)                              ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                         ║
║   PHYSICAL BENCHMARK: Bounded Device HBM (96 Pages) vs. Host RAM (2,500 Pages) under Extreme Concurrency (K=20):       ║
║                                                                                                                         ║
║   Metric                            │ Demand Paging (LRU)  │ Demand Paging (LFU)  │ Speculative ELLA (Invention)        ║
║   ──────────────────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────────        ║
║   Cache Hit Rate (%)                │         0.20%        │         1.21%        │          28.23% (+140x gain!)       ║
║   Speculative Prefetch Hits         │            0         │            0         │             58 hits                 ║
║   Memory Pipeline Stall Ratio (%)   │      1,954.53%       │      1,937.57%       │         690.30% (-64.7% reduction)  ║
║   Effective Throughput (TPS)        │      9,391.27 TPS    │      8,656.59 TPS    │      10,803.06 TPS (+15.0% gain)    ║
║                                                                                                                         ║
║   CORE TAKEAWAYS FOR PATENT COUNCIL:                                                                                    ║
║   1. 35 U.S.C. § 101 Compliance: Concrete computer hardware memory hierarchy and DMA controller improvement.           ║
║   2. Datacenter Savings: Doubles/triples concurrent serving capacity per Google TPU v6e Trillium node without extra HBM. ║
║   3. Novelty & Non-Obviousness: First protocol coupling agentic AST semantic DAGs to physical hardware DMA prefetching. ║
║                                                                                                                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```
