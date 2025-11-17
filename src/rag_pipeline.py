from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    # Fallback si langchain-huggingface n'est pas disponible
    HuggingFaceEndpoint = None
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import utils

load_dotenv(override=False)


class CulinaryRAG:
    """High-level helper that exposes ingest + query helpers to Streamlit."""

    def __init__(
        self,
        collection_name: str = "culinary_recipes",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        utils.ensure_directories()
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embed_model = os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm_model = os.getenv("OLLAMA_MODEL", "mistral")
        self.use_cloud_llm = os.getenv("USE_CLOUD_LLM", "false").lower() == "true"
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_model = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(utils.VECTOR_DIR),
            embedding_function=self.embeddings,
        )
        
        # Choix du LLM : cloud (HuggingFace) ou local (Ollama)
        if self.use_cloud_llm and self.hf_api_key and HuggingFaceEndpoint:
            self.llm = HuggingFaceEndpoint(
                repo_id=self.hf_model,
                huggingfacehub_api_token=self.hf_api_key,
                task="text-generation",
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                }
            )
        else:
            # Fallback sur Ollama local
            try:
                self.llm = Ollama(model=self.llm_model)
            except Exception as e:
                # Si Ollama n'est pas disponible, essayer HuggingFace
                if not self.hf_api_key or not HuggingFaceEndpoint:
                    raise RuntimeError(
                        f"Ollama n'est pas disponible ({e}) et HuggingFace n'est pas configuré. "
                        "Installez Ollama ou configurez USE_CLOUD_LLM=true et HUGGINGFACE_API_KEY."
                    )
                self.llm = HuggingFaceEndpoint(
                    repo_id=self.hf_model,
                    huggingfacehub_api_token=self.hf_api_key,
                    task="text-generation",
                )
        
        self.prompt = utils.chef_prompt_template()

    # ------------------------------------------------------------------ ingest
    def ingest_pdf(self, pdf_path: Path) -> int:
        """Load, chunk, and persist a PDF; returns number of chunks added."""
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata.setdefault("source", str(pdf_path))

        if not chunks:
            return 0

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        return len(chunks)

    def ingest_uploaded_file(self, uploaded_file: Any) -> int:
        """
        Persist an uploaded Streamlit file into data/recipes before indexing.
        """
        if uploaded_file is None:
            raise ValueError("Aucun fichier reçu.")

        filename = utils.sanitize_filename(getattr(uploaded_file, "name", "recipe.pdf"))
        target_path = utils.DATA_DIR / filename
        target_path.write_bytes(uploaded_file.getbuffer())
        return self.ingest_pdf(target_path)

    # ------------------------------------------------------------------ query
    def _build_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def _count_documents(self) -> int:
        return self.vectorstore._collection.count()  # type: ignore[attr-defined]

    def is_ready(self) -> bool:
        return self._count_documents() > 0

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        if not question.strip():
            raise ValueError("Merci de poser une question sur les recettes.")

        if not self.is_ready():
            raise RuntimeError(
                "Aucune recette indexée pour le moment. Merci d'importer un PDF."
            )

        # Récupérer les documents sources pour extraire le nom du fichier
        retriever = self._build_retriever(k=k)
        source_docs = retriever.get_relevant_documents(question)
        
        # Extraire le nom du fichier PDF depuis les métadonnées
        pdf_filename = "document inconnu"
        if source_docs:
            source_path = source_docs[0].metadata.get("source", "")
            if source_path:
                pdf_filename = Path(source_path).name
        
        # Créer un prompt personnalisé avec le nom du fichier
        custom_prompt = utils.chef_prompt_template_with_filename(pdf_filename)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
        )
        raw_response: Dict[str, Any] = qa_chain.invoke({"query": question})

        sources = [
            doc.metadata.get("source", "inconnu")
            for doc in raw_response.get("source_documents", [])
        ]
        answer = raw_response.get("result", "")
        return {"answer": answer, "sources": sources}


def bootstrap_with_pdf(pdf_name: str = "easy-chinese-recipes-pdf.pdf") -> Optional[int]:
    """
    Helper for CLI usage: quickly ingest the provided PDF sitting at repo root.
    """
    pdf_path = Path(__file__).resolve().parent.parent.parent / pdf_name
    if not pdf_path.exists():
        return None

    rag = CulinaryRAG()
    return rag.ingest_pdf(pdf_path)


if __name__ == "__main__":
    added = bootstrap_with_pdf()
    if added is None:
        print("❌ Aucun PDF easy_chinese_recipes trouvé à la racine du dépôt.")
    else:
        print(f"✅ {added} passages indexés depuis easy_chinese_recipes.")

