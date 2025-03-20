import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = "IMDB-Movie-Data.csv"
df = pd.read_csv(file_path)

df = df.fillna("Unknown").astype(str)  # Convert all columns to string

# Create combined text features for recommendations
df['combined_features'] = (
    df['Genre'] + " " +
    df['Description'] + " " +
    df['Director'] * 5 + " " +  # Give more weight to Directors
    df['Actors'] * 5             # Give more weight to Actors
)

# Convert text into numerical form using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# OMDb API Key (Replace with your actual key)
OMDB_API_KEY = "YOUR_OMDB_API_KEY"

def get_movie_poster(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={OMDB_API_KEY}"
    response = requests.get(url).json()
    
    if response.get("Response") == "True":
        return response.get("Poster")
    else:
        return None

# Recommendation Function
def recommend(movie_name, num_recommendations=5):
    movie_name = movie_name.lower()
    
    if movie_name not in df['Title'].str.lower().values:
        return ["Movie not found in the dataset. Please try another one."]
    
    idx = df[df['Title'].str.lower() == movie_name].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in scores[1:num_recommendations+1]]
    
    return df['Title'].iloc[movie_indices].tolist()

# Function to Show Selected Attributes in a Table
def show_combined_features(movie_name, selected_attributes):
    movie_name = movie_name.lower()
    
    if movie_name not in df['Title'].str.lower().values:
        return None
    
    movie_data = df.loc[df['Title'].str.lower() == movie_name, selected_attributes]
    return movie_data.T.reset_index().rename(columns={"index": "Attribute", 0: "Details"})

# Streamlit Web App UI
st.markdown(
    """
    <style>
        /* Background color */
        .stApp { background-color: #000000; }  /* Black Background */

        /* Title styling */
        h1 { 
            color: #FFD700 !important;  /* Gold */
            text-align: center;
            font-size: 40px;
        }

        /* Sidebar styling */
        .css-1d391kg { background-color: #1E1E1E !important; }  /* Dark Gray */

        /* Buttons */
        .stButton>button { 
            background-color: #FFD700 !important; /* Gold */
            color: #000000 !important;  /* Black */
            border-radius: 10px; 
            font-size: 16px; 
            padding: 10px 15px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #FFF8DC !important; /* Light Gold */
        }

        /* Dataframe Table */
        .dataframe th {
            background-color: #FFD700 !important;  /* Gold */
            color: #000000 !important;  /* Black */
        }

        /* Movie List Styling */
        .movie-list {
            font-size: 18px;
            color: #FFFFFF;  /* White */
            margin-left: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🍿 Movie Recommendation & Q&A System</h1>", unsafe_allow_html=True)

st.sidebar.header("📌 Select an option:")
option = st.sidebar.radio("", ["Movie Recommendations", "Ask About a Movie"])

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
                    st.image(poster_url, width=120)
                else:
                    st.write("📌 No poster found")

            with col2:
                st.write(f"🎬 **{movie}**")

elif option == "Ask About a Movie":
    st.subheader("❓ Ask About a Movie")
    movie_name = st.text_input("Enter the movie name:")
    
    selected_attributes = st.multiselect(
        "Select attributes to display:",
        options=df.columns[1:],  # Exclude "Title"
        default=["Genre", "Director", "Actors"]  # Default selected attributes
    )
    
    if st.button("Show Selected Attributes"):
        features_df = show_combined_features(movie_name, selected_attributes)
        
        if features_df is not None:
            st.write("🔹 **Movie Details in Table Format:**")
            st.dataframe(features_df)
        else:
            st.warning("⚠️ Movie not found in the dataset!")
