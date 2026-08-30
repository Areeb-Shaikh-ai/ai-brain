# Research Draft: Project ARIA Performance Analysis
**Target Conference:** MULTINOVA 2.0 (ICISEA 2026)
**Track:** Track 4 – Intelligent Systems & Architectures

---

## Title
**Empirical Analysis of Large Language Model Inference Latency and Resource Sensitivity on Quad-Core Mobile Architectures: A Case Study on Project ARIA**

## Abstract
This paper presents an empirical evaluation of localized Large Language Model (LLM) inference performance on consumer-grade, resource-constrained mobile hardware. Utilizing an 8th Generation Intel i5 quad-core processor and 16GB of DDR4 RAM, we analyze the execution of a 3-billion parameter model (Qwen 2.5) via 4-bit quantized CPU inference. Our experiments demonstrate a consistent baseline velocity of ~10.5 tokens per second (TPS). Crucially, the study identifies a "Switching Penalty," where minor peripheral interrupts resulted in a 45% decrease in inference speed. We further investigate the correlation between response length and thermal throttling on thin-and-light laptop architectures. These findings provide a benchmark for deploying sustainable, local-first AI assistants on legacy mobile workstations.

---

## 1. Introduction
The current AI landscape is dominated by cloud-based inference, which poses significant challenges regarding data privacy, latency, and operational costs. Project ARIA (Artificial Responsive Intelligence Assistant) explores the feasibility of "Local-First AI"—running sophisticated models entirely on-device. This study focuses on the architectural bottlenecks encountered when running a 3B parameter model on an Intel i5-8265U mobile CPU.

## 2. Experimental Methodology
### 2.1 Software Stack
*   **Inference Engine:** Ollama (based on llama.cpp)
*   **Model:** Qwen 2.5 3B (Quantized to 4-bit)
*   **Telemetry:** Custom Python-based benchmarking suite utilizing the `psutil` and `time` libraries.
*   **Environment:** Python 3.12.8 Stable build.

### 2.2 Hardware Configuration
*   **Machine:** Lenovo ThinkPad T490
*   **Processor:** Intel Core i5-8265U (1.60 GHz base, 4 Cores)
*   **Memory:** 16GB DDR4 RAM
*   **Storage:** NVMe SSD (D: Drive Project Root)

## 3. Results and Data Analysis
*Note: Data points gathered on August 29-30, 2026.*

| Metric | Clean State (Run 1) | With System Interrupt (Run 2) |
| :--- | :--- | :--- |
| **Inference Time (s)** | 16.27 s | 29.55 s |
| **Throughput (TPS)** | 10.56 TPS | 5.02 TPS |
| **Performance Delta** | Baseline | -45.1% |

### 3.1 Observations on Response Length
Our data reveals that total inference duration is linearly correlated with output length rather than input prompt complexity. Initial tests show a throughput drop from 10.5 TPS to 7.2 TPS during sustained generation, suggesting the onset of thermal throttling on the mobile quad-core architecture.

## 4. Discussion and Future Work
The results indicate that while 3B parameter models are viable for real-time interaction on legacy hardware, the "Inference Stability" is highly vulnerable to background OS tasks. Future work will involve implementing a "Context-Aware Cooling" logic and testing the efficiency of 1-bit (BitNet) architectures to further reduce the computational load on the CPU.

## 5. Conclusion
Project ARIA demonstrates that localized AI is achievable on 8th-generation mobile hardware. By understanding the hardware-software "handshake" and optimizing for CPU-bound constraints, we can bridge the gap between high-level research and accessible, private AI systems.

---
## References
[1] Vaswani, A., et al. (2017). "Attention Is All You Need."
[2] Gerganov, G. (2023). "llama.cpp: Port of Facebook's LLaMA model in C/C++."
[3] Alibaba Qwen Team. (2024). "Qwen 2.5: A Language Model Series."