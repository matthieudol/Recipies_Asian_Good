from __future__ import annotations

import streamlit as st

from src import utils
from src.rag_pipeline import CulinaryRAG


st.set_page_config(page_title="🍳 Culinary RAG", layout="wide")
st.title("🍳 Assistant Culinaire RAG")


@st.cache_resource(show_spinner=False)
def load_rag() -> CulinaryRAG:
    return CulinaryRAG()


rag = load_rag()

with st.sidebar:
    st.header("📚 Indexer des recettes")
    st.write("Charge un PDF (ex: easy_chinese_recipes) pour alimenter la base locale.")
    uploaded_file = st.file_uploader("Choisir un PDF", type=["pdf"])

    if uploaded_file and st.button("Indexer ce PDF"):
        with st.spinner("Indexation en cours..."):
            try:
                chunks = rag.ingest_uploaded_file(uploaded_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Impossible d'indexer le fichier: {exc}")
            else:
                st.success(f"✅ {chunks} passages ajoutés à la base.")

    st.divider()
    st.caption(
        "Astuce: lance `python -m src.rag_pipeline` pour pré-indexer "
        "le PDF easy_chinese_recipes depuis le terminal."
    )


st.subheader("🔍 Trouver ou adapter une recette")
query = st.text_area("Que souhaites-tu cuisiner aujourd'hui ?", height=120)

col1, col2 = st.columns([1, 1])
with col1:
    need_inspiration = st.button("Idée aléatoire")
with col2:
    run_query = st.button("Lancer la recherche", type="primary")

if need_inspiration:
    query = (
        "Propose une recette rapide à base de nouilles chinoises en ajoutant "
        "une option végétarienne et une version pour 6 personnes."
    )

if run_query:
    if not query.strip():
        st.warning("Commence par saisir une question (ingrédients, contraintes...).")
    else:
        with st.spinner("Le chef réfléchit..."):
            try:
                response = rag.query(query)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
            else:
                st.markdown("### 🍽️ Proposition du chef")
                st.write(response["answer"])
                st.markdown(utils.format_sources(response["sources"]))

