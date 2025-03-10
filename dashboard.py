import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the CSV file
df = pd.read_csv("Cleaned.csv")
df=df.sample(n=100,random_state=42)

# Preprocess text data
df["Skills"] = df["Skills"].str.lower().str.strip()
df["Education"] = df["Education"].str.lower().str.strip()

# Define the job description
job_description = {
    "Required Skills": "python, machine learning, data analysis",
    "Required Education": "bachelor's in computer science"
}

# Combine candidate skills and education
df["Combined"] = df["Skills"] + " " + df["Education"]

# Add the job description to the combined data
combined_data = df["Combined"].tolist() + [job_description["Required Skills"] + " " + job_description["Required Education"]]

# Fit the vectorizer on the combined data
tfidf = TfidfVectorizer()
tfidf.fit(combined_data)

# Transform candidate data
candidate_vectors = tfidf.transform(df["Combined"])

# Transform job description
job_vector = tfidf.transform([job_description["Required Skills"] + " " + job_description["Required Education"]])

# Calculate cosine similarity
similarity_scores = cosine_similarity(candidate_vectors, job_vector)
df["Similarity Score"] = similarity_scores.flatten()

# Rank candidates
ranked_df = df.sort_values(by="Similarity Score", ascending=False)

# Streamlit Dashboard
st.title("Candidate Ranking Dashboard")
st.write("This dashboard ranks candidates based on their similarity to the job description.")

# Display the job description
st.subheader("Job Description")
st.write(f"**Required Skills:** {job_description['Required Skills']}")
st.write(f"**Required Education:** {job_description['Required Education']}")

# Display the ranked candidates
st.subheader("Ranked Candidates")
st.dataframe(ranked_df[["Name", "Skills", "Education", "Similarity Score"]])

# Optionally, add a download button for the ranked candidates
st.download_button(
    label="Download Ranked Candidates as CSV",
    data=ranked_df.to_csv(index=False).encode('utf-8'),
    file_name='ranked_candidates.csv',
    mime='text/csv',
)
