"""
retrieval_module.py
Step 1 + Step 2: Systematic retrieval + evidence scoring + drug aggregation
PubMed + arXiv → deduplication → PRISMA → composite scoring → drug-level summary
"""

import re, numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import minmax_scale
from Bio import Entrez
import arxiv
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
Entrez.email = "nida.amir0083@gmail.com"   # REQUIRED for PubMed
CURRENT_YEAR = datetime.now().year

DISEASE_TERMS = [
    '"Anaplastic Thyroid Carcinoma"',
    '"Undifferentiated Thyroid Carcinoma"',
    'ATC',
    '"Thyroid Neoplasms"[MeSH]'
]

DEFAULT_DRUGS = [
    "Pazopanib", "Sorafenib", "Cabozantinib", "Everolimus"
]

DEFAULT_METHODS = [
    "drug repurposing", "drug repositioning",
    "knowledge graph", "hypothesis generation", "generative AI"
]

# Load BioBERT (for embeddings)
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# ---------------------------
# HELPERS
# ---------------------------

def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = re.sub(r'\s+', ' ', title).strip().lower()
    t = re.sub(r'[^\w\s]', '', t)
    return t

def embed_text(text):
    if not text:
        return np.zeros(768)
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def fetch_pubmed(query, max_results=200):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    pmids = record["IdList"]

    results = []
    if not pmids:
        return results

    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml")
    records = Entrez.read(handle)

    for article in records["PubmedArticle"]:
        try:
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]
            abstract = ""
            if "Abstract" in article["MedlineCitation"]["Article"]:
                abstract = " ".join([str(x) for x in article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]])
            pmid = article["MedlineCitation"]["PMID"]
            journal = article["MedlineCitation"]["Article"]["Journal"]["Title"]
            results.append({
                "source": "PubMed",
                "id": str(pmid),
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "published": None,
                "query": query
            })
        except:
            continue
    return results

def fetch_arxiv(query, max_results=100):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for r in search.results():
        results.append({
            "source": "arXiv",
            "id": r.entry_id.split("/")[-1],
            "title": r.title,
            "abstract": r.summary,
            "authors": [a.name for a in r.authors],
            "published": r.published.strftime("%Y-%m-%d"),
            "query": query
        })
    return results

def compute_features(row, disease_keywords, drug_keywords, query_embedding):
    text = (row["title"] or "") + " " + (row["abstract"] or "")
    text_low = text.lower()

    # Recency
    year = None
    if "published" in row and pd.notna(row["published"]):
        try:
            year = int(str(row["published"])[:4])
        except:
            year = None
    recency = (year - 2000) / (CURRENT_YEAR - 2000) if year and year > 2000 else 0.5

    # Mentions
    disease_mention = int(any(k in text_low for k in disease_keywords))
    drug_mention = int(any(k.lower() in text_low for k in drug_keywords))

    # Source weight
    source_weight = 1.0 if row["source"] == "PubMed" else 0.8

    # Embedding similarity
    emb = embed_text(text)
    sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)

    return pd.Series({
        "recency": recency,
        "disease_mention": disease_mention,
        "drug_mention": drug_mention,
        "source_weight": source_weight,
        "embedding_similarity": sim
    })

# ---------------------------
# MAIN PIPELINE
# ---------------------------

def run_retrieval_and_scoring(disease="Anaplastic Thyroid Carcinoma",
                              drugs=DEFAULT_DRUGS,
                              methods=DEFAULT_METHODS,
                              max_pubmed=200,
                              max_arxiv=100):

    # Build queries
    disease_block = " OR ".join([f'"{d}"' for d in [disease] + DISEASE_TERMS])
    drug_block = " OR ".join([f'"{d}"' for d in drugs])
    method_block = " OR ".join([f'"{m}"' for m in methods])

    pubmed_queries = [
        f"({disease_block}) AND ({drug_block}) AND ({method_block})",
        f"({disease_block}) AND ({drug_block})",
        f"({disease_block}) AND ({method_block})"
    ]

    arxiv_queries = [
        f'all:"drug repurposing" AND ({disease} OR anaplastic)',
        f'all:"knowledge graph" AND "drug repurposing"',
        f'all:"generative AI" AND "drug discovery"'
    ]

    # Run searches
    pubmed_results = []
    for q in pubmed_queries:
        pubmed_results.extend(fetch_pubmed(q, max_pubmed))

    arxiv_results = []
    for q in arxiv_queries:
        arxiv_results.extend(fetch_arxiv(q, max_arxiv))

    all_results = pubmed_results + arxiv_results
    df = pd.DataFrame(all_results)

    # Deduplicate
    before = len(df)
    df["norm_title"] = df["title"].apply(normalize_title)
    df = df.drop_duplicates(subset="norm_title").reset_index(drop=True)
    after = len(df)

    prisma_counts = {
        "PubMed retrieved": len(pubmed_results),
        "arXiv retrieved": len(arxiv_results),
        "Total retrieved": before,
        "Duplicates removed": before - after,
        "Final included": after
    }

    # Evidence scoring
    query_embedding = embed_text(f"{disease} drug repurposing")
    feats = df.apply(lambda r: compute_features(r, [disease.lower()] + [x.lower() for x in DISEASE_TERMS],
                                               drugs, query_embedding), axis=1)
    df = pd.concat([df, feats], axis=1)

    # Normalize + composite
    df["recency_norm"] = minmax_scale(df["recency"])
    df["embedding_norm"] = minmax_scale(df["embedding_similarity"])
    df["composite_score"] = (
        0.25 * df["recency_norm"] +
        0.20 * df["disease_mention"] +
        0.20 * df["drug_mention"] +
        0.15 * df["source_weight"] +
        0.20 * df["embedding_norm"]
    )

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df, prisma_counts

# ---------------------------
# DRUG-LEVEL AGGREGATION
# ---------------------------

def aggregate_drug_evidence(df, drugs=None):
    if drugs is None:
        drugs = DEFAULT_DRUGS

    drug_rows = []
    for drug in drugs:
        drug_lower = drug.lower()
        subset = df[df["title"].str.lower().str.contains(drug_lower) |
                    df["abstract"].str.lower().str.contains(drug_lower)]

        if len(subset) == 0:
            continue

        drug_rows.append({
            "Drug": drug,
            "Articles": len(subset),
            "Avg_Composite": round(subset["composite_score"].mean(), 3),
            "Max_Composite": round(subset["composite_score"].max(), 3),
            "Recency_Avg": round(subset["recency"].mean(), 3),
            "Avg_Similarity": round(subset["embedding_norm"].mean(), 3),
            "Disease_Mentions": int(subset["disease_mention"].sum())
        })

    df_drug = pd.DataFrame(drug_rows).sort_values("Avg_Composite", ascending=False).reset_index(drop=True)
    return df_drug

# ---------------------------
# PRISMA PLOT
# ---------------------------

def plot_prisma(prisma_counts):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis("off")

    steps = [
        f"PubMed retrieved: {prisma_counts['PubMed retrieved']}",
        f"arXiv retrieved: {prisma_counts['arXiv retrieved']}",
        f"Total retrieved: {prisma_counts['Total retrieved']}",
        f"Duplicates removed: {prisma_counts['Duplicates removed']}",
        f"Final included: {prisma_counts['Final included']}"
    ]

    y = 1.0
    for step in steps:
        ax.text(0.5, y, step, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", lw=1.5),
                fontsize=12)
        y -= 0.18

    arrow_y = 0.91
    for _ in range(len(steps)-1):
        ax.annotate("↓", xy=(0.5, arrow_y), xycoords="axes fraction",
                    ha="center", fontsize=14, weight="bold")
        arrow_y -= 0.18

    return fig

    


   
   

   

    
