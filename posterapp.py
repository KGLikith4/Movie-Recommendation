# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 00:31:26 2025

@author: likith
"""

import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load dataset
file_path = "IMDB-Movie-Data.csv"  # Ensure this file is in the same folder
df = pd.read_csv(file_path)

# ✅ Fill missing values
df = df.fillna("Unknown").astype(str)  # Convert all columns to string

# ✅ Create combined text features for recommendations
df['combined_features'] = (
    df['Genre'] + " " + 
    df['Description'] + " " +
    df['Director'] * 5 + " " +  # Give more weight to Directors
    df['Actors'] * 5             # Give more weight to Actors
)

# ✅ Convert text into numerical form using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# ✅ Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ✅ OMDb API Key (Replace with your actual key)
OMDB_API_KEY = "YOUR_OMDB_API_KEY"

# ✅ Function to Fetch Movie Posters
def get_movie_poster(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={OMDB_API_KEY}"
    response = requests.get(url).json()
    
    if response.get("Response") == "True":
        return response.get("Poster")  # Return poster URL
    else:
        return None  # No poster found

# ✅ Case-Insensitive Recommendation Function
def recommend(movie_name, num_recommendations=5):
    movie_name = movie_name.lower()  # Convert input to lowercase
    
    if movie_name not in df['Title'].str.lower().values:
        return ["Movie not found in the dataset. Please try another one."]
    
    idx = df[df['Title'].str.lower() == movie_name].index[0]  # Get movie index
    scores = list(enumerate(cosine_sim[idx]))  # Get similarity scores
    scores = sorted(scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    movie_indices = [i[0] for i in scores[1:num_recommendations+1]]  # Get top N movies
    
    return df['Title'].iloc[movie_indices].tolist()

# ✅ Case-Insensitive Q&A Function
def answer_question(movie_name, column_name):
    movie_name = movie_name.lower()  # Convert input to lowercase

    if movie_name not in df['Title'].str.lower().values:
        return "Movie not found in the dataset."
    
    if column_name not in df.columns:
        return "Invalid question. Please ask about available attributes."

    return f"{column_name} of '{movie_name.title()}': {df.loc[df['Title'].str.lower() == movie_name, column_name].values[0]}"

# 🎨 Custom CSS for Colors & Styling
st.markdown(
    """
    <style>
        /* Background color */
        .stApp { background-color: #0F0F0F; }

        /* Title styling */
        h1 { 
            color: #FFD700 !important; 
            text-align: center;
            font-size: 40px;
        }

        /* Sidebar background color */
        .css-1d391kg { background-color: #1E1E1E !important; }

        /* Buttons styling */
        .stButton>button { 
            background-color: #FF4500 !important; 
            color: white !important; 
            border-radius: 10px; 
            font-size: 16px; 
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FF6347 !important;
        }

        /* Movie list items */
        .movie-list {
            font-size: 18px;
            color: #FFD700;
            margin-left: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🎬 Streamlit Web App UI
st.markdown("<h1>🍿 Movie Recommendation & Q&A System</h1>", unsafe_allow_html=True)

# 🔹 Sidebar for navigation
st.sidebar.header("📌 Select an option:")
option = st.sidebar.radio("", ["Movie Recommendations", "Ask About a Movie"])

# 🔹 Movie Recommendations Section
if option == "Movie Recommendations":
    st.subheader("🔍 Get Movie Recommendations")
    movie_name = st.text_input("Enter a movie name:")
    
    if st.button("Recommend"):
        with st.spinner("🔍 Finding the best movies for you..."):
            recommendations = recommend(movie_name)
        st.success("✅ Done!")
        
        st.subheader(f"🎥 Movies similar to: {movie_name.title()}")
        
        for movie in recommendations:
            poster_url = get_movie_poster(movie)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if poster_url:
                    st.image(poster_url, width=120)  # Display poster
                else:
                    st.write("📌 No poster found")

            with col2:
                st.write(f"🎬 **{movie}**")  # Display movie name

# 🔹 Movie Q&A Section
elif option == "Ask About a Movie":
    st.subheader("❓ Ask About a Movie")
    movie_name = st.text_input("Enter the movie name:")
    column_name = st.selectbox("What do you want to know?", df.columns[1:])
    
    if st.button("Get Answer"):
        answer = answer_question(movie_name, column_name)
        st.write(answer)
