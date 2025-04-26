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

---
```
## 🧪 How it Works

1. Loads book descriptions and metadata from CSV
2. Splits and embeds the text into vectors using `HuggingFace MiniLM`
3. Stores embeddings in `ChromaDB` for semantic search
4. Takes a user query → finds top matches via cosine similarity
5. Filters based on:
   - 📘 Category
   - 😊 Emotional tone
6. Displays recommended books in a Gradio UI

---

## ▶️ Run Locally

### 1. Clone the repo

```
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
```

### 2. Set up virtual environment

```
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run the app

```> The Gradio UI will open in your browser automatically.```

## 🌍 Deployment Ideas

- Deploy to **HuggingFace Spaces**, **Render**, or **Railway**
- Wrap in a **FastAPI** backend for a more robust REST API
- Containerize with **Docker** for portability

---

## 📈 Future Improvements

- ✅ Save user preferences and history  
- 🔁 Add memory and chat-based interface using ChatGPT  
- 🎯 Personalize recommendations based on user profile  
- 📱 Build a mobile-first UI or React frontend  
- 🔊 Add voice input or image-based search  

---

## 📺 Based on FreeCodeCamp’s Project

Built while following this amazing LLM course by `freeCodeCamp`:

> [LLM Course – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio)](https://youtu.be/Q7mS1VHm3Yw)

---

## 🙌 Author

**Mohammad Moeenuddin**  
Passionate about building AI-first experiences with LLMs, LangChain, and ML infra.  
Say hi on [Twitter](https://x.com/im10Moin) | [LinkedIn](https://www.linkedin.com/in/mohammad-moeenuddin-558846226/) | [GitHub](https://github.com/Mohammad-10moin/)
---

## ⭐️ Show some love

If you found this project helpful:
⭐️ Star this repo
```It helps others discover this and supports open-source learning!```
---
