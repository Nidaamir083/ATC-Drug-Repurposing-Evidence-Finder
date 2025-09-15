import streamlit as st
import pandas as pd
from retrieval_module import run_retrieval_and_scoring, aggregate_drug_evidence, plot_prisma

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="ATC Drug Repurposing Finder",
    layout="wide"
)

# -------------------------
# CUSTOM BACKGROUND & STYLE
# -------------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.science.org/do/10.1126/science.adg6929/full/_20230317_on_ai_drugrepurposing.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.9);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------------
# HEADER BANNER
# -------------------------
st.image(
    "https://www.genengnews.com/wp-content/uploads/2023/04/GettyImages-Drug-Discovery.jpg",
    caption="AI for Drug Repurposing in Cancer",
    use_container_width=True
)

st.title("🔬 ATC Drug Repurposing Evidence Finder")

st.markdown("""
This app retrieves scientific articles (**PubMed + arXiv**), 
removes duplicates, applies **PRISMA-style filtering**, 
and computes **evidence scores** for candidate drugs in 
**Anaplastic Thyroid Carcinoma (ATC)**.

Inspired by: *Yan et al., 2024, npj Digital Medicine*.
""")

# -------------------------
# SIDEBAR LOGO
# -------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/44/Thyroid_cancer_ribbon.svg",
    caption="Thyroid Cancer Awareness",
    use_container_width=True
)

# -------------------------
# MAIN APP LOGIC
# -------------------------
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

    # PRISMA Counts
    st.subheader("📊 PRISMA Flow Counts")
    st.json(prisma_counts)

    # PRISMA Diagram
    st.subheader("📊 PRISMA Flow Diagram")
    fig = plot_prisma(prisma_counts)
    st.pyplot(fig)

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

   
