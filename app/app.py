import streamlit as st
import pandas as pd
import numpy as np
import random
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page title and layout
st.title("Stress Relief Recommender")
st.write("Select your mood and stress level to get personalized activity recommendations.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/stress_relief_data.csv')

df = load_data()

# Random Baseline
def random_baseline():
    activities = ['meditation', 'music', 'journaling', 'breathing', 'walking']
    return random.choice(activities)

# Popularity Baseline
def popularity_baseline(df):
    activity_scores = df.groupby('activity')['feedback'].sum().sort_values(ascending=False)
    return activity_scores.index[0]

# SVD Model
@st.cache_resource
def train_svd_model():
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[['user_id', 'activity', 'feedback']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=10, random_state=42)
    model.fit(trainset)
    return model

# SVD Recommendations
def recommend_svd(user_id, model, activities, n=3):
    predictions = [(activity, model.predict(user_id, activity).est) for activity in activities]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [activity for activity, _ in predictions[:n]]

# TF-IDF Content-Based Model
@st.cache_resource
def train_tfidf_model():
    df_liked = df[df['feedback'] == 1]
    df_liked['profile'] = df_liked['mood'] + ' ' + df_liked['activity']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_liked['profile'])
    return tfidf, tfidf_matrix, df_liked

# TF-IDF Recommendations
def content_based_recommend(mood, tfidf, tfidf_matrix, df_liked, n=3):
    mood_vector = tfidf.transform([mood])
    similarities = cosine_similarity(mood_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-n:][::-1]
    return df_liked.iloc[top_indices]['activity'].tolist()

# User input
mood = st.selectbox("Select your mood:", ["calm", "anxious"])
stress_level = st.selectbox("Select your stress level:", ["low", "high"])
user_id = st.number_input("Enter user ID (1-50):", min_value=1, max_value=50, value=1)

# Load models
svd_model = train_svd_model()
tfidf, tfidf_matrix, df_liked = train_tfidf_model()
activities = ['meditation', 'music', 'journaling', 'breathing', 'walking']

# Display recommendations
if st.button("Get Recommendations"):
    st.subheader("Recommendations")
    
    # Random Baseline
    st.write("**Random Baseline**")
    st.write(random_baseline())
    
    # Popularity Baseline
    st.write("**Popularity Baseline**")
    st.write(popularity_baseline(df))
    
    # SVD Recommendations
    st.write("**SVD (Collaborative Filtering)**")
    svd_recs = recommend_svd(user_id, svd_model, activities)
    for i, rec in enumerate(svd_recs, 1):
        st.write(f"{i}. {rec}")
    
    # TF-IDF Recommendations
    st.write("**TF-IDF (Content-Based)**")
    tfidf_recs = content_based_recommend(mood, tfidf, tfidf_matrix, df_liked)
    for i, rec in enumerate(tfidf_recs, 1):
        st.write(f"{i}. {rec}")

# Instructions for Git
# Save this file (Ctrl+S in VS Code)
# Commit to Git (run in terminal):
# git add app/app.py
# git commit -m "Implemented Streamlit UI for recommendations"
# git push