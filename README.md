## 🍳 Culinary RAG – Easy Chinese Recipes

MVP Streamlit app that runs a Retrieval-Augmented Generation (RAG) pipeline
on the provided `easy-chinese-recipes-pdf.pdf`. Ask cooking questions,
retrieve relevant passages, and let Mistral (via Ollama) craft playful answers.

### Project Structure
```
app/
├── app.py              # Streamlit UI
├── requirements.txt
├── src/
│   ├── rag_pipeline.py # LangChain RAG helper
│   └── utils.py        # Prompts + helpers
├── data/recipes/       # Persisted PDFs
└── vectorstore/        # Local Chroma DB
```

### 🚀 Getting Started
```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull mistral  # once
streamlit run app.py
```

### 🧠 Bootstrap the Knowledge Base
Index the default PDF from the repo root (run once):
```bash
cd app
python -m src.rag_pipeline
```
Or upload any recipe PDF via the Streamlit sidebar.

### 🧩 Tech Stack
- Streamlit UI
- LangChain (RetrievalQA chain)
- Chroma vector store (local persistence)
- HuggingFace MiniLM embeddings
- Ollama + Mistral LLM

### ✅ Next Steps / Ideas
- Add dietary filters (vegan, sans gluten)
- Estimate cooking/prep time from context
- Generate shopping lists & wine pairings
- Multi-language responses

Have fun building! 🎉

