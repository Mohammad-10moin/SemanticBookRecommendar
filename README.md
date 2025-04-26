# 📚 Semantic Book Recommendation System

A smart book recommender powered by **LangChain**, **HuggingFace Embeddings**, **ChromaDB**, and **Gradio**.  
Instead of relying on keywords, this system understands the **meaning** behind your query and recommends books that match your input *semantically*.

> "Looking for a story about forgiveness, second chances, or bittersweet endings? This app has you covered."

Live Link : (https://huggingface.co/spaces/10Moin/Semantic_Book_recommender)

---

## 🚀 Features

- 🔍 **Semantic Search** — Recommends books based on concepts, not just keywords
- 💡 **Emotional Filtering** — Sort results by tone: Happy, Sad, Surprising, Angry, Suspenseful
- 🎨 **Interactive Gradio UI** — Simple, clean interface to test recommendations
- ⚡ **Fast Retrieval** — Uses vector embeddings + ChromaDB for lightning-fast querying

---

## 🧠 Tech Stack

| Component         | Tool / Library                              |
|------------------|----------------------------------------------|
| Language Model    | [HuggingFace - MiniLM L6 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Framework         | [LangChain](https://www.langchain.com/)      |
| Vector Store      | [ChromaDB](https://www.trychroma.com/)       |
| UI                | [Gradio](https://www.gradio.app/)            |
| Data Processing   | Pandas, NumPy, Python standard libraries     |

---

## 📂 Project Structure

```bash
├── gradio-dashboard.py       # Main app
├── books_with_emotions.csv   # Books metadata + emotion tags
├── tagged_description.txt    # Text content for semantic embeddings
├── chroma_books_db/          # Persisted Chroma vector store
├── cached_documents.pkl      # Cached split documents
├── .env                      # Optional environment variables
└── README.md
