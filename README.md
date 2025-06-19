# HERBERT: Leveraging UMLS Hierarchical Knowledge to Enhance Clinical Entity Normalization in Spanish

## Overview

**HERBERT** is a BERT-based bi-encoder for medical entity normalization in Spanish, leveraging hierarchical knowledge from UMLS to enhance candidate retrieval in medical entity linking (MEL) pipelines.  
This repository contains the code, models, and resources to reproduce the results of the paper:


- **Task:** Normalization of disease, procedure, and symptom mentions to SNOMED-CT/UMLS codes  
- **Domain:** Spanish biomedical/clinical texts  
- **Benchmarks:** DisTEMIST, MedProcNER, SympTEMIST  
- **Key contributions:**  
  - Exploitation of hierarchical knowledge-graph structure in UMLS during candidate retrieval  
  - State-of-the-art results in top-25 accuracy across all corpora  
  - Generalizable methodology (language-agnostic)

---

## Features

- **Hierarchical Knowledge Integration:** Incorporates UMLS parent/ancestor relationships into the contrastive learning process.
- **Modular and Reproducible:** Scripts and code for training, inference, and evaluation.
- **FAISS-based Candidate Retrieval:** Efficient similarity search for large-scale medical vocabularies.
- **Evaluation Benchmarks:** Includes ready-to-use scripts for DisTEMIST, MedProcNER, and SympTEMIST corpora.
- **Notebooks and Examples:** Jupyter notebooks for step-by-step demo and analysis.
- **Language-agnostic:** Easily adaptable to other languages supported by UMLS.

---

## Installation

**Requirements (main):**
```plaintext
faiss-gpu==1.7.2
huggingface-hub==0.26.2
loguru==0.7.3
numpy==1.26.0
pandas==2.2.3
sentence-transformers==3.3.1
sentencepiece==0.2.0
tokenizers==0.21.0
torch==2.5.1
```
**(You can find the full requirements in `requirements.txt`.)**

Install dependencies with:

```bash
git clone https://github.com/ICB-UMA/HERBERT.git
cd HERBERT
pip install -r requirements.txt
```

## Quick Start

### 1. Download Pretrained Models

Model checkpoints will be available at [Hugging Face Model Hub](https://huggingface.co/ICB-UMA/) or in the [models/](models/) directory.