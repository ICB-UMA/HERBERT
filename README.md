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


### 2. Generate Triplets for Model Training

To prepare the data for training, you must generate triplets ({anchor, positive, negative}) using the UMLS knowledge graph.  
This is a **computationally intensive process** due to the size of the UMLS files and the need to build large-scale positive pairs for contrastive learning.

**Variants:**
- **ClinLinker:** Uses only synonym relationships (no hierarchy).
- **HERBERT-P:** Adds "parent" (hierarchical) relationships as positives.
- **HERBERT-GP:** Adds both "parent" and "grandparent" relationships as positives.

> *We strongly recommend running the notebook [`notebooks/triplets_definition.ipynb`](notebooks/triplets_definition.ipynb), which details the complete process for generating triplets for each model variant. This process may require several hours and substantial RAM, depending on hardware and UMLS file size.*

You can configure:
- **Number of positive pairs per anchor** (e.g., 15 or 30)
- **Relationship depth** (synonyms only, parents, or parents+grandparents)

### 3. Train the Model Using SapBERT's Repository

Once you have generated the triplets file with positive and negative pairs, you can train a HERBERT or ClinLinker variant using the [SapBERT repository](https://github.com/cambridgeltl/sapbert).  
Your generated `.txt` file with positive pairs serves as the **pretraining data** for SapBERT-style contrastive learning.

**Steps:**
1. **Clone the SapBERT repository:**
    ```bash
    git clone https://github.com/cambridgeltl/sapbert.git
    cd sapbert
    ```

2. **Prepare your pretraining data:**
    - Use the path to your generated pairs file (from step 2).

3. **Edit and execute the pretraining script:**
    - Update the `TRAIN_PATH` variable in `train/pretrain.sh` to point to your triplets file.
    - Then run:
    ```bash
    bash train/pretrain.sh
    ```
    - This script will launch the model pretraining using your custom triplets.

> For additional configuration (batch size, model name, epochs, etc.), refer to the [SapBERT documentation](https://github.com/cambridgeltl/sapbert).

**Note:** The triplets file you generated is fully compatible with SapBERT's pretraining pipeline.

### 4. Run Inference Using FAISS Encoder

Once the model is trained, you can use it for fast candidate retrieval on Spanish clinical corpora (such as DisTEMIST, MedProcNER, and SympTEMIST) with the provided FAISS-based encoder.

Below is an example of how to use `src/faiss_encoder.py` for building the FAISS index and running inference:

```python
import faiss_encoder as faiss_enc
from utils import load_corpus_data
import pandas as pd
from metrics import calculate_topk_accuracy

DATA_PATH = "YOUR_PATH"
corpus = "DisTEMIST"  # or "MedProcNER" / "SympTEMIST"
f_type = "FlatIP"
max_length = 256

# Load the gold standard, train, and gazetteer data for your corpus
gs_df, train_df, gaz_df = load_corpus_data(DATA_PATH, corpus)

# Combine train and gazetteer terms/codes into a single dataframe for indexing
train_gaz_df = pd.concat([train_df[["term", "code"]], gaz_df[["term", "code"]]], ignore_index=True)

# Build and fit the FAISS encoder using the trained HERBERT-P model
faiss_encoder = faiss_enc.FaissEncoder("ICB-UMA/HERBERT-P", f_type, max_length, train_gaz_df)
faiss_encoder.fit_faiss()

# Get candidates with concept similarity
gs_df["candidates"], gs_df["codes"], _ = faiss_encoder.get_candidates(gs_df["term"].tolist(), k=200)

calculate_topk_accuracy(gs_preds, top_k_values)
```
* ICB-UMA/HERBERT-P should point to your trained model path or Hugging Face Hub model.
* f_type is the FAISS index type (e.g., "FlatIP" for inner product search).
* You can now use faiss_encoder to efficiently retrieve the most similar candidates for any entity mention in your corpus.

*You could replace train_gaz_df with your own ontology (SNOMED-CT, UMLS, etc). See `src/faiss_encoder.py` for more options and usage details.*