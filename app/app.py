import streamlit as st
import pandas as pd
import numpy as np
import random
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

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

# LinUCB Bandit Implementation
class LinUCBBandit:
    def __init__(self, arms, context_dim, alpha=1.0):
        self.arms = arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = {arm: np.identity(context_dim) for arm in arms}
        self.b = {arm: np.zeros(context_dim) for arm in arms}

    def select_arm(self, context):
        ucb_values = {}
        for arm in self.arms:
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])
            ucb = np.dot(theta, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
            ucb_values[arm] = ucb
        return max(ucb_values, key=ucb_values.get)

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

# Train Bandit
@st.cache_resource
def train_bandit():
    activities = ['meditation', 'music', 'journaling', 'breathing', 'walking']
    context_dim = 4  # 2 for mood (anxious, calm), 2 for stress_level (high, low)
    bandit = LinUCBBandit(activities, context_dim, alpha=1.0)
    # Train on historical data
    np.random.seed(42)
    for _, row in df.iterrows():
        context = get_context(row['mood'], row['stress_level'])
        arm = row['activity']
        reward = row['feedback']
        bandit.update(arm, context, reward)
    return bandit

# Context for Bandit
def get_context(mood, stress_level):
    mood_vec = [1 if mood == 'anxious' else 0, 1 if mood == 'calm' else 0]
    stress_vec = [1 if stress_level == 'high' else 0, 1 if stress_level == 'low' else 0]
    return np.array(mood_vec + stress_vec)

# Feedback file
feedback_file = 'data/feedback.csv'
if not os.path.exists(feedback_file):
    pd.DataFrame(columns=['user_id', 'mood', 'stress_level', 'activity', 'feedback']).to_csv(feedback_file, index=False)

# User input
mood = st.selectbox("Select your mood:", ["calm", "anxious"])
stress_level = st.selectbox("Select your stress level:", ["low", "high"])
user_id = st.number_input("Enter user ID (1-50):", min_value=1, max_value=50, value=1)

# Load models
svd_model = train_svd_model()
tfidf, tfidf_matrix, df_liked = train_tfidf_model()
bandit = train_bandit()
activities = ['meditation', 'music', 'journaling', 'breathing', 'walking']

# Display recommendations
if st.button("Get Recommendations"):
    st.subheader("Recommendations")
    
    # Random Baseline
    random_rec = random_baseline()
    st.write("**Random Baseline**")
    st.write(random_rec)
    
    # Popularity Baseline
    pop_rec = popularity_baseline(df)
    st.write("**Popularity Baseline**")
    st.write(pop_rec)
    
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
    
    # Bandit Recommendation
    st.write("**LinUCB Bandit**")
    context = get_context(mood, stress_level)
    bandit_rec = bandit.select_arm(context)
    st.write(bandit_rec)

    # Real-time feedback
    st.subheader("Provide Feedback")
    feedback_activity = st.selectbox("Select the activity you tried:", activities + [random_rec, pop_rec, bandit_rec] + svd_recs + tfidf_recs, index=0)
    feedback_value = st.radio("Did you like it?", (1, 0))
    if st.button("Submit Feedback"):
        new_feedback = pd.DataFrame({
            'user_id': [user_id],
            'mood': [mood],
            'stress_level': [stress_level],
            'activity': [feedback_activity],
            'feedback': [feedback_value]
        })
        new_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
        # Update bandit in real-time
        context = get_context(mood, stress_level)
        bandit.update(feedback_activity, context, feedback_value)
        st.success("Feedback submitted! Bandit updated.")
        # Reload data for other models (optional, as retraining may be heavy)
        # global df
        df = pd.concat([df, new_feedback], ignore_index=True)
        st.cache_data.clear()
        st.cache_resource.clear()

# Instructions for Git
# Save this file (Ctrl+S in VS Code)
# Commit to Git (run in terminal):
# git add app/app.py
# git commit -m "Added real-time feedback to Streamlit UI"
# git push