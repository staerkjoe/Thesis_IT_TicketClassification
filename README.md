# Distilling GPT-5 into On-Premise Student Models for IT Ticket Classification

> **Master's Thesis**, Copenhagen Business School, 2026  
> Johannes & Alexander  
> Supervised by Raghava Rao Mukkamala

## TL;DR
* **The Problem:** Deploying high-accuracy text classification on an unlabelled, newly introduced ServiceNow taxonomy (the cold-start problem) under strict local hardware constraints, without routing data via third-party cloud APIs.
* **The Solution:** A two-stage knowledge distillation framework. 
  * **Stage I (Label Creation):** Employing GPT-5 offline to construct a "silver" training corpus of pseudo-labels and diagnostic reasoning chains. 
  * **Stage II (Model Distillation):** Utilizing these teacher-generated traces to **QLoRA fine-tune** two compact, open-weight student architectures (LLaMA 3.1 8B and Ministral 8B) for entirely local deployment.
* **The Business Impact:** Achieved **84% accuracy** (matching the teacher), secured **100% data sovereignty**, and reduced per-ticket inference costs by **92%** ($0.00015 vs $0.0018).

### Core Technologies & Topics

* **Core Methodology:** `Knowledge Distillation` • `Rationale Distillation` • `Confidence-Informed Self-Consistency (CISC)` • `Text Classification`
* **LLMs & Architectures:** `LLaMA 3.1 8B` • `Ministral 8B` • `GPT-5 (Teacher baseline)`
* **Fine-Tuning & PEFT:** `QLoRA` • `LoRA` • `4-bit Quantization (NF4)` • `Supervised Fine-Tuning (SFT)`
* **Infrastructure & MLOps:** `Azure Machine Learning Studio` • `Weights & Biases (W&B)` • `Poetry`
* **Optimization Frameworks:** `Unsloth` • `Hugging Face PEFT` • `TRL` • `PyTorch`
---

## Overview

Enterprise IT support teams handle thousands of support tickets monthly, each requiring manual routing to the correct assignment group. Large language models (LLMs) offer strong classification capability, but deploying them in regulated enterprise environments runs into three simultaneous constraints: **GDPR data sovereignty** (ticket text cannot leave the infrastructure boundary), **no labelled training data** on newly introduced platforms, and **hardware budgets** that preclude running a 100B+ parameter model locally.

This thesis investigates whether **black-box knowledge distillation** from a proprietary teacher LLM into a compact, locally deployable student model can resolve all three constraints at once.

---

## The Pipeline

<img width="1000" height="1000" alt="MethodologyOverview" src="https://github.com/user-attachments/assets/9c1bd2ee-4c71-4e0d-8ec3-6e46f3a6a17f" />



**The pipeline operates in two stages:**

### Stage I: Silver Label Construction

GPT-5 acts as a teacher, generating structured pseudo-labels for **26,331 raw IT support tickets** from Nuuday's YouSee DAWN platform (ServiceNow). To ensure label quality without human annotation, we use **Confidence-Informed Self-Consistency (CISC)**:

- 5 independent Chain-of-Thought reasoning paths per ticket
- Confidence-weighted majority voting
- A composite **Scientific Confidence Score (S\*)** per ticket
- Filtering to **18,461 high-confidence samples** (S\* ≥ 0.75)

Three labelling strategies are benchmarked against a 50-ticket human-adjudicated gold standard (Cohen's κ = 0.434). S\* is shown to be a statistically significant predictor of external label validity (Spearman ρ = 0.31–0.41, p < 0.05).

### Stage II: Knowledge Distillation via QLoRA

Two open-weight decoder models are fine-tuned on the silver corpus using **QLoRA**, updating only ~0.9% of parameters on a single NVIDIA A100 GPU:

| Student Model | Parameters updated | Training time | Full-label agreement |
|---|---|---|---|
| LLaMA 3.1-8B Instruct | ~42M / 0.9% | 3.9 h | **84.14%** |
| Ministral-8B Instruct | ~42M / 0.9% | 18.5 h | **84.30%** |

Both models exceed the primary fidelity target of ≥80% agreement with the GPT-5 teacher.

---

## Key Results

### 1. Distillation Fidelity & Hierarchical Degradation

<img width="3853" height="1753" alt="hierarchical_accuracy_four_models" src="https://github.com/user-attachments/assets/184fe589-2ef8-4ac5-97cf-8b9f13d4c000" />

| Metric | LLaMA Vanilla | Mistral Vanilla | LLaMA FT | Mistral FT |
|---|---|---|---|---|
| **Full-label agreement** | 0.11% | 0.00% | **84.14%** | **84.30%** |
| **Weighted F1** | 0.17% | 0.00% | 83.73% | 83.82% |
| **Macro F1** | 0.05% | 0.00% | 65.89% | 63.75% |
| **L1 accuracy** | — | — | 97% | 97% |
| **L2 accuracy** | — | — | 95% | 95% |
| **ECE (Calibration) ↓** | 0.780 | 0.861 | **0.193** | **0.183** |

**Key Insights:**
* **Zero-Shot Baseline Collapse:** Out-of-the-box vanilla models fail completely (~0% full-label match), proving that foundational models cannot parse proprietary corporate taxonomies without target adaptation.
* **Fidelity Breakthrough:** QLoRA fine-tuning achieves a massive performance leap, yielding **84.14%** (LLaMA) and **84.30%** (Ministral) exact multi-level label agreement with the GPT-5 teacher.
* **Hierarchical Resilience:** Structural accuracy remains exceptionally robust at the top tiers of categorization (**97% at Level 1** and **95% at Level 2**), with gentle degradation occurring only at the highly granular Level 3 scenario tier.
* **Uncertainty Calibration:** Expected Calibration Error (ECE) plummets from ~0.80 down to **0.18**. Fine-tuning effectively aligns the models' output confidence scores with their true empirical accuracy rates.

---

### 2. Computational & Architectural Trade-offs

| Dimension | LLaMA FT | Mistral FT | Winner |
|---|---|---|---|
| **Weighted F1 ↑** | 83.73% | **83.82%** | Mistral (+0.09%) |
| **Macro F1 ↑** | **65.89%** | 63.75% | LLaMA (+2.14%) |
| **Mean latency ↓** | **732 ms** | 1,885 ms | LLaMA (2.5x faster) |
| **Training time ↓** | **3.9 h** | 18.5 h | LLaMA (4.7x faster) |
| **Peak VRAM ↓** | **11.9 GB** | 15.3 GB | LLaMA (-22%) |
| **Training cost ↓** | **$15.60** | $74.00 | LLaMA (79% savings) |

**Key Insights:**
* **The Accuracy Parity:** While Mistral claims a nominal, statistically negligible win on Weighted F1 (+0.09%), LLaMA exhibits superior class-imbalance resilience, outperforming Mistral on Macro F1 by **2.14%**.
* **Production Bottlenecks:** LLaMA completely outclasses Ministral in production deployment viability. LLaMA processes tickets **2.5x faster** at inference time and drastically condenses the model development loop with a **3.9-hour training run** compared to Mistral's 18.5 hours.
* **Hardware Economy:** Driven by architectural optimizations like Grouped-Query Attention (GQA), LLaMA maintains a lean memory footprint of **11.9 GB Peak VRAM**. This allows the model to scale comfortably within low-cost enterprise hardware limits (such as a ubiquitous 16GB NVIDIA T4 GPU).

---

### 3. Error Typology & Reasoning Transfer

<img width="3552" height="1754" alt="error_typology (1)" src="https://github.com/user-attachments/assets/9658240e-22fb-4e32-ab34-87656d986fda" />

**Key Insights:**
* **Symptom Ambiguity Congruence:** The student error profile is overwhelmingly dominated by **Level-3 Symptom Confusion** (~200 individual instances for both models). 
* **Minimal Catastrophic Failures:** Crucially, severe errors, such as Level-1 Cross-Domain misclassifications (e.g., routing a billing failure into a core network outage bucket), are heavily suppressed.
* **Empirical Proof of Rationale Internalization:** The fine-tuned students mirror the exact disagreement signatures found among senior human experts (who initially clashed on 54% of tickets due to inherent taxonomy overlapping). This strongly suggests the student models successfully internalized the teacher's **diagnostic reasoning logic** rather than blindly memorizing superficial text patterns.

### Reasoning Transfer Evidence

Error typology on the silver set mirrors human inter-rater disagreement patterns (Level-3 symptom confusion dominant at 67–69%), interpreted as evidence of **genuine reasoning transfer** rather than label memorisation.

---

## Demo

The system is deployed as a **Streamlit application** that accepts raw ticket text and returns a predicted three-level taxonomy tag with a confidence score and full reasoning chain.

[![Video-Demo](https://github.com/user-attachments/assets/36b2aab1-cedb-495a-8039-b7f41476404c)](https://drive.google.com/file/d/1VqHKWYDKzweJKIyBGffWPFMBv4AJ2h8p/view?usp=sharing)


```
Input:  "Support - Login/My YouSee/Dawn Native Customer (non migrated)/Other The Customer has, in connection with moving address, 
        received a new TV package and now needs to create a new login, 
        but the system does not allow him to do so as it remembers him from his old login. 
        Therefore, the Customer cannot access his YouSee Play 40 package at all.

Category: Login/Mit YouSee/Dawn Native Customer (non migrated)/Other

loginPoint: App   loginWith: Username YAC-Resolution"

Output: Tag: (1g Self service, 2g Mit YouSee), 3 Login issues
        Confidence: 0.9999130753469105
        Reasoning: "The customer cannot create/access a new My YouSee login after moving address due to the system retaining an old login, 
        which blocks access to their YouSee Play 40 package. 
        This is a self-service My YouSee authentication problem rather than a TV content issue."
```

---

## Deployment Recommendation

**Recommended Model: LLaMA 3.1-8B (Fine-tuned)**
Selected as the official production system because it uniquely satisfies Nuuday's three binding constraints:

* **GDPR Compliance:** 100% on-premise inference completely eliminates the data exposure risks of external APIs.
* **Cost & Hardware:** Operates on standard T4 enterprise GPUs (11.87 GB VRAM). Inference costs just **$0.000152 per ticket** (92% reduction compared to the GPT-5 API).
* **Operational Independence:** Rapid 3.9-hour retraining cycle frees Nuuday from vendor pricing shifts and model deprecations.

### Alternative Architectures
* **Ministral 8B (Fine-tuned):** *Not recommended.* Fails hardware constraints by requiring an expensive A100 GPU, resulting in a per-ticket cost ($0.002095) higher than the baseline API.
* **GPT-5 (PoC API):** *Benchmark only.* Fails GDPR constraints for production use but is retained as the baseline human-evaluation benchmark (49% accuracy ceiling).

LLaMA's $15.60 training cost breaks even against continued API usage in approximately **1.8 months** at Nuuday's observed ticket volume (~8,777 tickets/month).

<img width="1185" height="769" alt="fig_breakeven_training_5k (1)" src="https://github.com/user-attachments/assets/c5d0eba6-1376-4122-b5a8-1077be29cba8" />


---

## Green AI

The pipeline updates 0.9% of model parameters at a compute cost two orders of magnitude below conventional full fine-tuning. Strubell et al. (2019) estimated that training a single large Transformer with neural architecture search can emit up to 284 metric tons CO₂e. The QLoRA adapter training in this project represents a negligible fraction of that footprint without sacrificing classification capability.

---

## Repository Structure

```
├── data/
│   ├── raw/                                    # Raw ServiceNow exports (not included — PII)
│   ├── silver/                                 # Teacher-labelled silver dataset (not included — PII)
│   └── gold/                                   # 50-ticket expert-labelled gold standard (not included — PII)
│
├── config/
│   ├── model_config_base.yaml                  # Shared A100 production defaults
│   ├── model_config_llama.yaml                 # Llama-3.1-8B fine-tuning overrides
│   ├── model_config_llama_vanilla.yaml         # Llama-3.1-8B vanilla baseline (eval only)
│   ├── model_config_mistral.yaml               # Ministral-8B fine-tuning overrides
│   ├── model_config_mistral_vanilla.yaml       # Ministral-8B vanilla baseline (eval only)
│   ├── pipeline_test_config.yaml               # Fast end-to-end smoke-test overrides
│   ├── prompt_template_teacher.yaml            # GPT-5 teacher prompt + label taxonomy
│   └── prompt_template_student.yaml            # Student SFT prompt template
│
├── utils/
│   ├── preprocessing/
│   │   └── PII_detection.py                    # Presidio-based anonymisation pipeline
│   │   └── data_handler.py                    
│   ├── tagging/
│   │   └── label_creation.py             
│   │   └── label_creation_cisc.py              # CISC teacher labelling pipeline (GPT-5 / self-consistency)
│   │   └── label_creation_dynamic_cisc.py      # RAG-CISC teacher labelling pipeline
│   ├── training/
│   │   ├── llm_handler.py                      # Model loading, LoRA setup, prompt formatting
│   │   └── wandb_plots.py                      # Training callbacks, loss curves, W&B logging
│   └── evaluation/
│       └── wandb_eval.py                       # Full eval pipeline (silver / fresh / gold metrics)
│
├── scripts/
│   ├── train.py                                # Stage II distillation fine-tuning entry point
│   ├── eval.py                                 # Post-training evaluation (any dataset type)
│   └── eval_posttrain.py                       # Resume eval from existing W&B run + checkpoint
│
├── app.py                                      # Interactive demo application
│
├── notebooks/
│   ├── EDA & Preprocessing/
│   ├── tagging/
│   ├── Results Analysis/
│
├── pyproject.toml                              # Poetry dependency manifest
├── poetry.lock                                 # Pinned dependency lockfile
├── .pre-commit-config.yaml                      
├── .env                                        # API keys and secrets (not included)
└── .gitignore
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/staerkjoe/Thesis_IT_TicketClassification
cd Thesis_IT_TicketClassification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit demo
streamlit run app/streamlit_app.py
```

**Hardware requirements:**
- Training: NVIDIA A100 80GB (or equivalent ≥40GB VRAM with QLoRA)
- Inference (LLaMA): NVIDIA T4 16GB or equivalent
- Inference (Mistral): NVIDIA A10 24GB or equivalent

---

## References

Key references used in this work:

- Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*
- Hsieh et al. (2023) — *Distilling Step-by-Step*
- Wang et al. (2022) — *Self-Consistency Improves Chain of Thought Reasoning*
- Taubenfeld et al. (2025) — *Confidence Improves Self-Consistency*
- Zhou et al. (2023) — *LIMA: Less Is More for Alignment*
- Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
- Schwartz et al. (2019) — *Green AI*
- Strubell et al. (2019) — *Energy and Policy Considerations for Deep Learning in NLP*

---

*Data used in this thesis is proprietary to Nuuday A/S and cannot be publicly released.*
