# 🎮 Game Recommender System

An interactive, content-based recommendation engine built with Python and Streamlit. This application suggests video games based on player tags, genres, and developer metadata.

## 🧠 How It Works
Instead of relying on basic keyword matching, this engine uses **Natural Language Processing (NLP)**. 
1. **Data Engineering:** Raw Steam data was cleaned, deduplicated, and combined into a descriptive "metadata soup".
2. **Vectorization:** Used `TfidfVectorizer` to penalize overly common words (like "action") and highlight unique identifiers (like "cyberpunk" or "souls-like").
3. **Similarity Mapping:** Calculated the mathematical distance between games using **Cosine Similarity** to generate confidence-scored recommendations.

## 🛠️ Tech Stack
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn (TF-IDF, Cosine Similarity)
* **Frontend UI:** Streamlit

## 🚀 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `python -m streamlit run app.py`
