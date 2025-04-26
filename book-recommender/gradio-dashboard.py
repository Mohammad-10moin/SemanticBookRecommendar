import os
import pickle
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

# === LangChain and Chroma imports ===
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # <- using new maintained import

# === Load environment variables ===
load_dotenv()

# === Constants ===
CACHE_FILE = "cached_documents.pkl"
CHROMA_DB_DIR = "chroma_books_db"
TEXT_FILE_PATH = "tagged_description.txt"

# === Load books data ===
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"
books['large_thumbnail'] = np.where(
    books['thumbnail'].isna(),
    "cover-not-found.jpg",
    books['large_thumbnail']
)

# === Function to load or cache documents ===
def load_or_cache_documents(path: str, cache_path: str = CACHE_FILE):
    if os.path.exists(cache_path):
        print("🔄 Loading cached documents...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        print("🆕 Loading and splitting raw documents...")
        raw_documents = TextLoader(path, encoding="utf-8").load()
        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
        documents = text_splitter.split_documents(raw_documents)
        with open(cache_path, "wb") as f:
            pickle.dump(documents, f)
        return documents

# === Load or cache the documents ===
documents = load_or_cache_documents(TEXT_FILE_PATH)

# === Initialize HuggingFace Embeddings ===
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load or persist Chroma vector DB ===
if os.path.exists(CHROMA_DB_DIR):
    print("🔁 Loading existing Chroma vector store...")
    db_books = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=huggingface_embeddings
    )
else:
    print("🧠 Creating new Chroma vector store...")
    db_books = Chroma.from_documents(
        documents,
        embedding=huggingface_embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    db_books.persist()

# === Semantic Recommendation Logic ===
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# === Format results for display ===
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# === Gradio UI ===
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

# === Launch app ===
if __name__ == "__main__":
    dashboard.launch(share=True)
