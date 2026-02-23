# 🎬 NLP-Driven Content-Based Movie Recommendation System

## 📌 Project Overview

The **NLP-Driven Content-Based Movie Recommendation System** is a machine learning project that recommends movies based on content similarity using Natural Language Processing (NLP).

Unlike collaborative filtering systems that depend on user ratings, this model analyzes movie metadata such as overview, genres, keywords, cast, and crew to find movies that are contextually similar.

This project demonstrates practical implementation of NLP, feature engineering, and similarity-based recommendation techniques.

---

## 🚀 Key Features

- ✅ Content-based recommendation system
- ✅ No dependency on user ratings
- ✅ NLP-powered text processing
- ✅ TF-IDF / Count Vectorization
- ✅ Cosine similarity scoring
- ✅ Clean and modular implementation
- ✅ Easily extendable for deployment

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **NLTK (optional for preprocessing)**
- **Jupyter Notebook**

---



---

## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset**, which includes detailed metadata for approximately 5000 movies.

Dataset contains:
- Title
- Genres
- Keywords
- Overview
- Cast
- Crew
- Movie ID

The movies and credits datasets are merged to create a unified dataset for feature engineering.

---

## 🧠 Methodology

### 1️⃣ Data Loading & Merging

```python
import pandas as pd

movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")

movies = movies.merge(credits, on="title")
```

---

### 2️⃣ Feature Engineering

Selected important columns:
- genres
- keywords
- overview
- cast
- crew

These features are combined into a single column called **tags**.

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```

---

### 3️⃣ Text Vectorization

Convert text into numerical format using TF-IDF or CountVectorizer:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies['tags'])
```

---

### 4️⃣ Similarity Calculation

Compute cosine similarity between movie vectors:

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```

---






## 📈 Future Improvements

- Add movie posters using TMDB API
- Deploy using Streamlit or Flask
- Improve embeddings using Word2Vec or BERT
- Add user-based personalization layer

---

## 🎓 Learning Outcomes

- Understanding NLP text preprocessing
- Feature engineering techniques
- TF-IDF & vector space models
- Cosine similarity implementation
- Building real-world recommendation systems

---

## 👨‍💻 Author

**Rohit Sahu**  
Machine Learning & NLP Enthusiast  

GitHub: https://github.com/Solitaryseeker  
LinkedIn: https://linkedin.com/in/rohit-sahu-7142742a7  

---

⭐ If you like this project, consider giving it a star!
