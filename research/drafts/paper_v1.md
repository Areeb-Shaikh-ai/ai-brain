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
Experiments conducted on August 31, 2026, compared two parameter scales of the Qwen 2.5 family.

### 3.1 Inference Throughput (Table 1)
| Model Scale | Peak Throughput (TPS) | Avg. Speed | Latency (Long Prompt) |
| :--- | :--- | :--- | :--- |
| **Qwen 2.5 3B** | 10.78 TPS | ~8.2 TPS | 67.79 s |
| **Qwen 2.5 0.5B** | 41.18 TPS | ~31.9 TPS | 11.56 s |

### 3.2 Findings
The 0.5B parameter model demonstrated a 3.8x increase in average throughput compared to the 3B model. However, initial observation suggests a qualitative "Intelligence Gap" in complex reasoning tasks. On-device AI on quad-core mobile CPUs (i5-8265U) is highly viable at the <1B parameter scale, offering "instant" response times (<12s) even for long-form outputs exceeding 1,500 characters.

### 3.3 Scaling and Thermal Observations
Our data confirms that total inference duration is governed by output length (token generation) rather than input prompt complexity. 

Key Observations:
1. **The Warm-up Effect:** In the 3B model, throughput improved from 6.23 TPS to 10.78 TPS as output length increased from 725 to 2,923 characters. This suggests a "computational momentum" where the mobile CPU stabilizes after the initial Time-to-First-Token (TTFT) latency.
2. **Thermal Constraints:** During sustained generation (Prompt 3, 3B model), the T490 chassis reached peak operating temperature. While the model maintained ~10 TPS for this burst, previous experiments show a degradation to ~7 TPS in sustained sessions exceeding 120 seconds, identifying a clear thermal ceiling for on-device AI on thin-and-light hardware.

## 4. Discussion and Future Work
The results indicate that while 3B parameter models are viable for real-time interaction on legacy hardware, the "Inference Stability" is highly vulnerable to background OS tasks. Future work will involve implementing a "Context-Aware Cooling" logic and testing the efficiency of 1-bit (BitNet) architectures to further reduce the computational load on the CPU.

## 5. Conclusion
Project ARIA demonstrates that localized AI is achievable on 8th-generation mobile hardware. By understanding the hardware-software "handshake" and optimizing for CPU-bound constraints, we can bridge the gap between high-level research and accessible, private AI systems.

---
## References
[1] Vaswani, A., et al. (2017). "Attention Is All You Need."
[2] Gerganov, G. (2023). "llama.cpp: Port of Facebook's LLaMA model in C/C++."
[3] Alibaba Qwen Team. (2024). "Qwen 2.5: A Language Model Series."