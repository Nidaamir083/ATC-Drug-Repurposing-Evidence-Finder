# ATC-Drug-Repurposing-Evidence-Finder
Retrieve articles from popular research article sites PubMed and ArXiv
# 🔬 ATC Drug Repurposing Evidence Finder

This project implements **Step 1 & 2** of a systematic **drug repurposing pipeline** 
for **Anaplastic Thyroid Carcinoma (ATC)**, inspired by Yan et al. (2024, *npj Digital Medicine*).

## 🚀 Features
- Retrieves articles from **PubMed + arXiv**
- Deduplicates results with **normalized titles**
- Generates **PRISMA-style flow counts**
- Computes **evidence scores** based on:
  - Recency
  - Disease mention
  - Drug mention
  - Source weight (PubMed > arXiv)
  - Semantic similarity (BioBERT embeddings)
- Aggregates results into a **Drug-Level Evidence Table**
- Interactive **Streamlit app** with CSV downloads

## 📂 Project Structure
ATC-Drug-Repurposing
│── app.py # Streamlit frontend
│── retrieval_module.py # Retrieval & evidence scoring
│── requirements.txt # Dependencies
│── README.md # Documentation
│── .gitignore # Ignore cache & env files


## ⚡ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
