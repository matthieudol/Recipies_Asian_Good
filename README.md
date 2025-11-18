# 🍳 Culinary RAG – Easy Chinese Recipes

Assistant culinaire basé sur Retrieval-Augmented Generation (RAG) qui explore le PDF `easy-chinese-recipes-pdf.pdf`, propose des recettes adaptées (végétarien, nombre de convives, contraintes diététiques) et affiche les sources utilisées. Le projet suit les exigences du document `Project guidelines.pdf` : pipeline RAG complet, UI Streamlit, documentation, déploiement cloud et explication des choix techniques.

---

## 1. Architecture Globale

```
User ── Streamlit UI (app.py)
          │
          ├─ Upload / indexation PDF
          └─ Questions en langage naturel
                │
                ▼
        LangChain RAG Pipeline (src/rag_pipeline.py)
          ├─ PyPDFLoader + RecursiveCharacterTextSplitter
          ├─ HuggingFace MiniLM embeddings
          ├─ Chroma (persist_directory=vectorstore/)
          └─ LLM Mistral via Ollama (mode local) ou HuggingFace API (mode cloud)
```

- **Mode local (par défaut)** : Ollama + Mistral 7B, aucune dépendance cloud.
- **Mode cloud** : branche `cloud-experiments` qui active une version simplifiée du pipeline (`USE_SIMPLE_RAG=true`) et interroge HuggingFace Inference.

---

## 2. Installation Locale

### Prérequis
- Python 3.10+
- Git
- [Ollama](https://ollama.ai/) + modèle `mistral`
- macOS / Linux (testé sur macOS Sonoma)

### Étapes
```bash
git clone https://github.com/matthieudol/Recipies_Asian_Good.git
cd Recipies_Asian_Good/app
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
ollama pull mistral

# Indexer le PDF fourni
python -m src.rag_pipeline         # enregistre les embeddings dans vectorstore/

# Lancer l’interface
streamlit run app.py
```

> L’application est disponible sur `http://localhost:8501`.  
> Les dossiers `data/recipes/` et `vectorstore/` sont créés automatiquement.

---

## 3. Utilisation

1. **Indexer un PDF**  
   - via le terminal (`python -m src.rag_pipeline`)  
   - ou via la sidebar Streamlit (Upload + bouton “Indexer ce PDF”).

2. **Poser une question**  
   - ex. “Je veux des nouilles sautées version végétarienne pour 6 personnes”.

3. **Lire la réponse**  
   - Format imposé par le prompt :  
     ```
     📄 Document analysé: easy-chinese-recipes-pdf.pdf
     ✅ Recette directe trouvée: …
     Conseils pratiques…
     ```
   - Les sources utilisées sont listées sous la réponse.

---

## 4. Variables d’Environnement

| Contexte | Variable | Description |
|----------|----------|-------------|
| Local | `EMBED_MODEL` | défaut `sentence-transformers/all-MiniLM-L6-v2` |
| Local | `OLLAMA_MODEL` | défaut `mistral` |
| Cloud | `HUGGINGFACE_API_KEY` | token HuggingFace |
| Cloud | `HUGGINGFACE_MODEL` | ex. `mistralai/Mistral-7B-Instruct-v0.2` |
| Cloud | `USE_CLOUD_LLM` | `true` pour forcer HuggingFace |
| Cloud | `USE_SIMPLE_RAG` | `true` (pipeline simplifié) |

⚠️ Sur Streamlit Cloud, préférer les **Environment Variables** plutôt que `secrets.toml` pour éviter les erreurs de parsing. Un template est disponible dans `DEPLOYMENT.md`.

---

## 5. Déploiement Streamlit Cloud (Résumé)

1. Pousser la version cloud-ready (`cloud-experiments` ou une branche dédiée`).  
2. Lancer [share.streamlit.io](https://share.streamlit.io), relier le repo.  
3. Paramètres → Environment variables :  
   ```
   HUGGINGFACE_API_KEY = "hf_xxx"
   HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
   USE_CLOUD_LLM = "true"
   USE_SIMPLE_RAG = "true"
   ```
4. `Main file path` = `app/app.py`.  
5. Après déploiement, uploader `easy-chinese-recipes-pdf.pdf` via la sidebar.

Voir `DEPLOYMENT.md` pour le pas-à-pas détaillé (captures d’écran, gestion des secrets).

---

## 6. Choix Techniques vs Project Guidelines

| Exigence (guidelines.pdf) | Réalisation |
|---------------------------|-------------|
| Pipeline RAG documenté | README + architecture décrite ci-dessus |
| Données persistées localement | `vectorstore/` (Chroma) + `data/recipes/` |
| UI Streamlit ergonomique | Sidebar upload, zone de saisie, affichage des sources |
| Adaptations culinaires intelligentes | Prompt Michelin avec sorties “Recette directe / Adaptation possible” |
| Gestion des sources | chaque chunk conserve `metadata["source"]` |
| Documentation de déploiement | `DEPLOYMENT.md` + section 5 de ce README |
| Bonus / idées futures | Section 8 ci-dessous |

---

## 7. Détails du Pipeline RAG

1. **Ingestion**  
   - `PyPDFLoader` → pages  
   - `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap)  
   - Ajout du chemin source dans `metadata["source"]`.

2. **Indexation**  
   - Embeddings via `HuggingFaceEmbeddings` (MiniLM)  
   - Stockage dans `Chroma` (persist_directory = `vectorstore/`).

3. **Retrieval + Génération**  
   - `vectorstore.as_retriever(k=5)`  
   - `RetrievalQA.from_chain_type` avec prompt custom (`src/utils.py`)  
   - LLM = Ollama Mistral (local) ou wrapper HuggingFace API (cloud).

4. **Contraintes du prompt**  
   - Mention obligatoire “📄 Document analysé: …”  
   - Distinction Recette directe / Adaptation possible / Aucune recette  
   - Conseils pratiques + touche fun pour répondre aux guidelines UX.

---

## 8. Améliorations & Roadmap

- [ ] Migrer vers `langchain_huggingface` / `langchain_chroma` (versions non dépréciées).  
- [ ] Filtres diététiques (vegan, halal, sans gluten).  
- [ ] Estimation temps de préparation + liste de courses.  
- [ ] Suggestions accords mets-vins.  
- [ ] Support multilingue (FR / EN / ES).  
- [ ] Monitoring RAG (LangFuse) + tests d’intégration automatiques.  
- [ ] Mode “Batch indexing” pour plusieurs PDFs.

---

## 9. Structure du Repo
```
RAG_Recipes/
├── README.md                   # documentation globale
├── Project guidelines.pdf      # cahier des charges du cours
├── easy-chinese-recipes-pdf.pdf
└── app/
    ├── app.py                  # UI Streamlit
    ├── requirements.txt
    ├── src/
    │   ├── rag_pipeline.py     # pipeline principal (Ollama + HuggingFace fallback)
    │   └── utils.py            # prompts, helpers
    ├── data/recipes/           # fichiers uploadés
    └── vectorstore/            # base Chroma persistée
```

---

## 10. Crédits & Licence

- Projet académique réalisé dans le cadre du Master 2 “Introduction à la Génération d’IA incluant RAG”.  
- PDF source : `easy-chinese-recipes-pdf.pdf` (fourni dans le repo).  
- Licence : usage éducatif uniquement.

Enjoy & bon appétit ! 🍜
