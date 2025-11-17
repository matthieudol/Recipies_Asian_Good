from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.prompts import PromptTemplate


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "recipes"
VECTOR_DIR = Path(__file__).resolve().parent.parent / "vectorstore"


def ensure_directories() -> None:
    """Create default data/vector directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-friendly version of an uploaded filename."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    return safe.strip("_") or "recipe.pdf"


def chef_prompt_template() -> PromptTemplate:
    """Prompt template used by the RetrievalQA chain (version par défaut)."""
    return chef_prompt_template_with_filename("document")

def chef_prompt_template_with_filename(pdf_filename: str) -> PromptTemplate:
    """Prompt template with explicit PDF filename to avoid LLM inventing names."""
    template = textwrap.dedent(
        f"""
        Tu es un chef étoilé Michelin spécialisé dans les recettes chinoises faciles.
        
        RÈGLES STRICTES:
        - Utilise UNIQUEMENT les informations fournies dans le contexte ci-dessous
        - N'invente JAMAIS de noms de recettes, de sites web, ou de références qui ne sont pas dans le contexte
        - Si le contexte ne contient pas d'information sur la question, dis clairement: "Je n'ai pas trouvé cette information dans les recettes disponibles"
        - Ne suggère PAS de consulter des sites web externes ou des catégories non mentionnées
        
        IMPORTANT: Le document analysé s'appelle exactement: "{pdf_filename}"
        Ne change PAS ce nom. Utilise-le tel quel dans ta réponse.
        
        Contexte disponible:
        {{context}}

        Question de l'utilisateur:
        {{question}}

        Réponds en structurant ta réponse ainsi:
        
        Commence TOUJOURS par indiquer: "📄 Document analysé: {pdf_filename}"
        
        Si tu trouves une recette DIRECTE dans le contexte qui correspond à la question:
        - Commence par: "✅ Recette directe trouvée: [nom de la recette]"
        - Liste les ingrédients principaux et les étapes clés
        
        Si tu ne trouves PAS de recette directe mais que tu peux ADAPTER une recette existante:
        - Commence par: "🔄 Adaptation possible: Le RAG n'a pas trouvé de recette directe à base de [ingrédient/plat demandé], mais je vous propose une recette adaptée de: [nom de la recette source]"
        - Explique comment adapter la recette (remplacements d'ingrédients, modifications)
        - Liste les ingrédients adaptés et les étapes modifiées
        
        Si aucune recette pertinente n'est trouvée dans le contexte:
        - Dis clairement: "❌ Aucune recette pertinente trouvée dans les documents disponibles"
        - N'invente pas d'informations
        
        Dans tous les cas, termine par:
        - Des conseils pratiques basés sur le contexte
        - Une suggestion fun (emoji autorisés)
        """
    ).strip()
    return PromptTemplate(template=template, input_variables=["context", "question"])


def format_sources(sources: Sequence[str]) -> str:
    """Return a readable bullet list of retrieved sources for Streamlit."""
    if not sources:
        return "Aucune source trouvée dans la base actuelle."

    formatted = "\n".join(f"- {Path(src).name}" for src in sources)
    return f"### Sources\n{formatted}"


def humanize_chunks(chunks: Iterable[str], max_chars: int = 500) -> str:
    """Preview the first retrieved chunks (for debugging/trace display)."""
    preview = []
    total = 0
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        total += len(chunk)
        preview.append(chunk)
        if total >= max_chars:
            break
    return "\n---\n".join(preview)

