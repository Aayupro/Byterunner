import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the CSV file
@st.cache_data  # Cache the data to improve performance
def load_data():
    return pd.read_csv("Cleaned.csv")

df = load_data()

# Check if "Project Count" exists and rename it to "Experience"
if "Projects Count" in df.columns:
    df.rename(columns={"Projects Count": "Experience"}, inplace=True)
else:
    st.warning("Column 'Projects Count' not found. Using default experience values.")
    df["Experience"] = 0  # Default experience value if column is missing

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

# Define Education Score (higher education = higher score)
education_score_map = {
    "high school": 1,
    "bachelor's": 2,
    "master's": 3,
    "phd": 4
}
df["Education Score"] = df["Education"].map(education_score_map).fillna(0)

# Define Experience Score (more projects = higher experience)
# Normalize the Experience column to a 0-1 range
if "Experience" in df.columns:
    df["Experience Score"] = df["Experience"] / df["Experience"].max()
else:
    df["Experience Score"] = 0  # Default experience score if column is missing

# Calculate Weighted Score
df["Weighted Score"] = (
    0.5 * df["Similarity Score"] +  # Similarity contributes 50%
    0.3 * df["Experience Score"] +  # Experience contributes 30%
    0.2 * df["Education Score"]     # Education contributes 20%
)

# Rank candidates by Weighted Score
ranked_df = df.sort_values(by="Weighted Score", ascending=False)

# Streamlit Dashboard
st.title("Candidate Ranking Dashboard")
st.write("This dashboard ranks candidates based on their similarity to the job description.")

# Display the job description
st.subheader("Job Description")
st.write(f"**Required Skills:** {job_description['Required Skills']}")
st.write(f"**Required Education:** {job_description['Required Education']}")

# Display the ranked candidates
st.subheader("Ranked Candidates")

# Check which columns exist and only include them in the display
columns_to_display = ["Name", "Skills", "Education", "Experience", "Similarity Score", "Weighted Score"]
existing_columns = [col for col in columns_to_display if col in ranked_df.columns]

if not existing_columns:
    st.error("No valid columns found to display.")
else:
    st.dataframe(ranked_df[existing_columns])

# Optionally, add a download button for the ranked candidates
st.download_button(
    label="Download Ranked Candidates as CSV",
    data=ranked_df.to_csv(index=False).encode('utf-8'),
    file_name='ranked_candidates.csv',
    mime='text/csv',
)
