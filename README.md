# Distilling GPT-5 into On-Premise Student Models for IT Ticket Classification

> **Master's Thesis** — Copenhagen Business School, 2026  
> Johannes & Alexander  
> Supervised by Raghava Rao Mukkamala

---

## Overview

Enterprise IT support teams handle thousands of support tickets monthly, each requiring manual routing to the correct assignment group. Large language models (LLMs) offer strong classification capability — but deploying them in regulated enterprise environments runs into three simultaneous constraints: **GDPR data sovereignty** (ticket text cannot leave the infrastructure boundary), **no labelled training data** on newly introduced platforms, and **hardware budgets** that preclude running a 100B+ parameter model locally.

This thesis investigates whether **black-box knowledge distillation** from a proprietary teacher LLM into a compact, locally deployable student model can resolve all three constraints at once.

---

## The Pipeline

<img width="665" height="642" alt="MethodologyOverview (1)" src="https://github.com/user-attachments/assets/fd96b41d-3543-4cf6-8ff3-c31033022cc0" />


The pipeline operates in two stages:

### Stage I — Silver Label Construction

GPT-5 acts as a teacher, generating structured pseudo-labels for **26,331 raw IT support tickets** from Nuuday's YouSee DAWN platform (ServiceNow). To ensure label quality without human annotation, we use **Confidence-Informed Self-Consistency (CISC)**:

- 5 independent Chain-of-Thought reasoning paths per ticket
- Confidence-weighted majority voting
- A composite **Scientific Confidence Score (S\*)** per ticket
- Filtering to **18,461 high-confidence samples** (S\* ≥ 0.75)

Three labelling strategies are benchmarked against a 50-ticket human-adjudicated gold standard (Cohen's κ = 0.434). S\* is shown to be a statistically significant predictor of external label validity (Spearman ρ = 0.31–0.41, p < 0.05).

### Stage II — Knowledge Distillation via QLoRA

Two open-weight decoder models are fine-tuned on the silver corpus using **QLoRA** — updating only ~0.9% of parameters on a single NVIDIA A100 GPU:

| Student Model | Parameters updated | Training time | Full-label agreement |
|---|---|---|---|
| LLaMA 3.1-8B Instruct | ~42M / 0.9% | 3.9 h | **84.14%** |
| Ministral-8B Instruct | ~42M / 0.9% | 18.5 h | **84.30%** |

Both models exceed the primary fidelity target of ≥80% agreement with the GPT-5 teacher.

---

## Key Results

<img width="3853" height="1753" alt="hierarchical_accuracy_four_models" src="https://github.com/user-attachments/assets/184fe589-2ef8-4ac5-97cf-8b9f13d4c000" />


### Distillation Fidelity (Silver Evaluation Set, N = 1,847)

| Metric | LLaMA Vanilla | Mistral Vanilla | LLaMA FT | Mistral FT |
|---|---|---|---|---|
| Full-label agreement | 0.11% | 0.00% | **84.14%** | **84.30%** |
| Weighted F1 | 0.17% | 0.00% | 83.73% | 83.82% |
| Macro F1 | 0.05% | 0.00% | 65.89% | 63.75% |
| L1 accuracy | — | — | 97% | 97% |
| L2 accuracy | — | — | 95% | 95% |
| ECE | 0.780 | 0.861 | 0.193 | 0.183 |

### Architecture Trade-off

| Dimension | LLaMA FT | Mistral FT | Winner |
|---|---|---|---|
| Weighted F1 ↑ | 83.73% | **83.82%** | Mistral |
| Macro F1 ↑ | **65.89%** | 63.75% | LLaMA |
| Mean latency ↓ | **732 ms** | 1,885 ms | LLaMA |
| Training time ↓ | **3.9 h** | 18.5 h | LLaMA |
| Peak VRAM ↓ | **11.9 GB** | 15.3 GB | LLaMA |
| Training cost ↓ | **$15.60** | $74.00 | LLaMA |

<img width="3552" height="1754" alt="error_typology (1)" src="https://github.com/user-attachments/assets/9658240e-22fb-4e32-ab34-87656d986fda" />


### Reasoning Transfer Evidence

Error typology on the silver set mirrors human inter-rater disagreement patterns (Level-3 symptom confusion dominant at 67–69%), interpreted as evidence of **genuine reasoning transfer** rather than label memorisation.

---

## Demo

The system is deployed as a **Streamlit application** that accepts raw ticket text and returns a predicted three-level taxonomy tag with a confidence score and full reasoning chain.

[![Video-Demo](https://github.com/user-attachments/assets/36b2aab1-cedb-495a-8039-b7f41476404c)](https://drive.google.com/file/d/1VqHKWYDKzweJKIyBGffWPFMBv4AJ2h8p/view?usp=sharing)


```
Input:  "Support - Login/My YouSee/Dawn Native Customer (non migrated)/Other The Customer has, in connection with moving address, received a new TV package and now needs to create a new login, but the system does not allow him to do so as it remembers him from his old login. Therefore, the Customer cannot access his YouSee Play 40 package at all.  Category: Login/Mit YouSee/Dawn Native Customer (non migrated)/Other   loginPoint: App   loginWith: Username YAC-Resolution"
Output: Tag: (1g Self service, 2g Mit YouSee), 3 Login issues
        Confidence: 0.9999130753469105
        Reasoning: "The customer cannot create/access a new My YouSee login after moving address due to the system retaining an old login, which blocks access to their YouSee Play 40 package. 
        This is a self-service My YouSee authentication problem rather than a TV content issue."
```

---

## Deployment Recommendation

**Weighted deployment scoring matrix.** Scores are on a 1–5 scale; ↑ = higher is better, ↓ = lower is better. Weighted totals rounded to two decimal places.

| **Criterion** | **Weight** | **GPT-5 PoC** | **LLaMA FT** | **Ministral FT** |
| :--- | :--- | :---: | :---: | :---: |
| Operational classification quality ↑ | 25% | 4 | 3 | 3 |
| Data privacy & sovereignty ↑ | 20% | 2 | 5 | 5 |
| Inference latency ↑ | 15% | 5 | 4 | 2 |
| Infrastructure cost ↑ | 15% | 3 | 5 | 2 |
| Long-term maintainability ↑ | 15% | 2 | 5 | 4 |
| Deployment complexity ↓ | 10% | 5 | 3 | 3 |
| **Weighted total** | | **3.30** | **4.05** | **3.15** |

*Scoring guide:* 5 = excellent, 4 = good, 3 = adequate, 2 = poor, 1 = unacceptable. For ↓ criteria, a score of 5 means the system is the *least* burdensome on that dimension. 
*Classification quality note:* PoC and student models are scored against different reference sets (human gold vs. teacher silver respectively); see Section 8.4.

A weighted multi-criteria scoring matrix across six dimensions (classification quality, data privacy, latency, cost, maintainability, deployment complexity) yields:

| System | Weighted Score |
|---|---|
| **LLaMA 3.1-8B Fine-tuned** | **4.05** ✅ Recommended |
| GPT-5 PoC | 3.30 |
| Mistral Fine-tuned | 3.15 |

LLaMA's $15.60 training cost breaks even against continued API usage in approximately **1.8 months** at Nuuday's observed ticket volume (~8,777 tickets/month).

<img width="1185" height="769" alt="fig_breakeven_training_5k (1)" src="https://github.com/user-attachments/assets/c5d0eba6-1376-4122-b5a8-1077be29cba8" />


---

## Green AI

The pipeline updates 0.9% of model parameters at a compute cost two orders of magnitude below conventional full fine-tuning. Strubell et al. (2019) estimated that training a single large Transformer with neural architecture search can emit up to 284 metric tons CO₂e. The QLoRA adapter training in this project represents a negligible fraction of that footprint — without sacrificing classification capability.

---

## Repository Structure

```
├── data/
│   ├── raw/                    # Raw ServiceNow exports (not included — PII)
│   └── silver/                 # Filtered silver dataset (not included — PII)
├── src/
│   ├── data_handler.py         # Data loading, cleaning, consolidation
│   ├── PII_detection.py        # Presidio-based anonymisation pipeline
│   ├── label_creation_cisc.py  # CISC teacher labelling pipeline
│   ├── config_labels.yaml      # Prompt templates and taxonomy config
│   ├── train_qlora.py          # QLoRA fine-tuning (LLaMA / Mistral)
│   └── evaluate.py             # Evaluation framework
├── app/
│   └── streamlit_app.py        # Demo application
├── notebooks/
│   └── EDA.ipynb               # Exploratory data analysis
└── thesis/
    └── *.tex                   # LaTeX source files
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

*Data used in this thesis is proprietary to Nuuday A/S and cannot be publicly released. Model weights are available on request subject to Nuuday's data governance approval.*
