import streamlit as st
import pandas as pd
from retrieval_module import run_retrieval_and_scoring, aggregate_drug_evidence

st.set_page_config(page_title="ATC Drug Repurposing Finder", layout="wide")

st.title("🔬 ATC Drug Repurposing Evidence Finder")

st.markdown("""
This app retrieves scientific articles (PubMed + arXiv), 
deduplicates results, applies **PRISMA-style filtering**, 
and computes **evidence scores** for candidate drugs in 
**Anaplastic Thyroid Carcinoma (ATC)**.
""")

# Inputs
disease = st.text_input("Disease:", "Anaplastic Thyroid Carcinoma")
drugs = st.text_area("Candidate Drugs (comma-separated):",
                     "Pazopanib, Sorafenib, Cabozantinib, Everolimus")
methods = st.text_area("Method Keywords:",
                       "drug repurposing, drug repositioning, knowledge graph, generative AI")

if st.button("🔎 Run Search"):
    with st.spinner("Fetching and scoring articles..."):
        drug_list = [d.strip() for d in drugs.split(",")]
        method_list = [m.strip() for m in methods.split(",")]

        df, prisma_counts = run_retrieval_and_scoring(disease, drug_list, method_list)
        df_drug = aggregate_drug_evidence(df, drug_list)

    st.success("Done ✅")

    # PRISMA
    st.subheader("📊 PRISMA Flow Counts")
    st.json(prisma_counts)

    # Article Evidence
    st.subheader("📑 Top Evidence Articles")
    st.dataframe(df[["title", "source", "composite_score"]].head(20))

    # Drug Evidence
    st.subheader("💊 Drug-Level Evidence Summary")
    st.dataframe(df_drug)

    # Downloads
    st.download_button("⬇️ Download Article Evidence Table",
                       data=df.to_csv(index=False),
                       file_name="evidence_table.csv")

    st.download_button("⬇️ Download Drug Evidence Table",
                       data=df_drug.to_csv(index=False),
                       file_name="drug_evidence_table.csv")
