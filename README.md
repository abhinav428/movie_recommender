# 🎬 Movie Recommendation System

A content-based movie recommender using metadata from the **TMDB 5000 Movies Dataset**.
It recommends movies based on similarity between:
- Genres
- Keywords
- Overview text
- Top Cast members
- Director names

---

## 🔗 Live Demo

Streamlit App: https://movie-recommender-on.streamlit.app/
---

## 🚀 Features
✔ Uses **TF-IDF Vectorizer** for text representation  
✔ **Cosine Similarity** for recommendation rankings  
✔ Supports fuzzy title matching (e.g. “Mission Impossible” → “Mission: Impossible”)  
✔ Built in a clean and modular format for easy future upgrades  

---

## 📂 Dataset

TMDB 5000 Movies + Credits Dataset  
🔗 Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata  
Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the same directory as the notebook before running.

---

## 🧠 Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn (TF-IDF + Cosine Similarity)
- NLTK (optional text preprocessing)
- difflib (fuzzy matching)
- Streamlit
---

## ▶️ How to Run

Open in Google Colab or Jupyter Notebook:

```bash
pip install -r requirements.txt
