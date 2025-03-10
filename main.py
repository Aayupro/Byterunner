import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --------------------- Step 1: Extract and Structure Job Descriptions ---------------------

# Load job descriptions and titles
job_desc_df = pd.read_csv("job_descriptions.csv")
job_titles_df = pd.read_csv("job_titles.csv")

# Function to extract job details
def extract_job_details(description):
    details = {
        "Job Title": None,
        "Required Skills": None,
        "Salary Range": None,
        "Experience": None,
        "Benefits": None,
        "Job Type": None,
    }

    # Extract Salary Range
    salary_match = re.search(r"Salary: (₹[\d,]+\.\d+ - ₹[\d,]+\.\d+)", description)
    if salary_match:
        details["Salary Range"] = salary_match.group(1)

    # Extract Experience
    exp_match = re.search(r"Experience: ([\w\s]+) \(Preferred\)", description)
    if exp_match:
        details["Experience"] = exp_match.group(1)

    # Extract Benefits
    benefits_match = re.search(r"Benefits: ([\w\s,]+) Schedule:", description)
    if benefits_match:
        details["Benefits"] = benefits_match.group(1)

    # Extract Job Type
    type_match = re.search(r"Job Types: ([\w\s,-]+) Salary:", description)
    if type_match:
        details["Job Type"] = type_match.group(1)

    # Extract Required Skills
    skills_match = re.search(r"We are looking for hire experts ([\w\s]+)\.", description)
    if skills_match:
        details["Required Skills"] = skills_match.group(1)

    return details

# Apply the function to each job description
job_desc_df["Details"] = job_desc_df["Job Description"].apply(extract_job_details)
job_desc_df = pd.concat([job_desc_df.drop(["Details"], axis=1), job_desc_df["Details"].apply(pd.Series)], axis=1)

# Add job titles
job_desc_df["Job Title"] = job_titles_df["Job Title"]

# Create structured DataFrame
structured_jobs_df = job_desc_df[["Job Title", "Required Skills", "Salary Range", "Experience", "Benefits", "Job Type"]]

# Save structured job descriptions to CSV
structured_jobs_df.to_csv("structured_job_descriptions.csv", index=False)

# --------------------- Step 2: Load and Preprocess Candidate Data ---------------------

# Load the candidate data
candidates_df = pd.read_csv("Cleaned.csv")

# Preprocess candidate text data
candidates_df["Skills"] = candidates_df["Skills"].str.lower().str.strip()
candidates_df["Education"] = candidates_df["Education"].str.lower().str.strip()
candidates_df["Combined"] = candidates_df["Skills"] + " " + candidates_df["Education"]

# --------------------- Step 3: Match Candidates to Job Descriptions ---------------------

# Combine candidate and job description data for fitting the vectorizer
combined_data = candidates_df["Combined"].tolist() + structured_jobs_df["Required Skills"].tolist()

# Fit the vectorizer on the combined data
tfidf = TfidfVectorizer()
tfidf.fit(combined_data)

# Transform candidate data
candidate_vectors = tfidf.transform(candidates_df["Combined"])

# Loop through each job description
for index, job in structured_jobs_df.iterrows():
    # Transform job description
    job_vector = tfidf.transform([job["Required Skills"]])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(candidate_vectors, job_vector)
    candidates_df["Similarity Score"] = similarity_scores.flatten()

    # Rank candidates
    ranked_df = candidates_df.sort_values(by="Similarity Score", ascending=False)

    # Filter candidates based on salary range
    min_salary = float(job["Salary Range"].split(" - ")[0].replace("₹", "").replace(",", ""))
    max_salary = float(job["Salary Range"].split(" - ")[1].replace("₹", "").replace(",", ""))
    filtered_df = ranked_df[(ranked_df["Salary Expectation"] >= min_salary) & (ranked_df["Salary Expectation"] <= max_salary)]

    # Save results to a CSV file
    job_title = job["Job Title"].replace("/", "_")  # Sanitize job title for filename
    filtered_df.to_csv(f"ranked_candidates_{job_title}.csv", index=False)

    # Visualize results
    fig = px.bar(filtered_df, x="Name", y="Similarity Score", title=f"Ranked Candidates for {job['Job Title']}")
    fig.show()

# --------------------- Step 4: Drop Sensitive Columns ---------------------

# Drop sensitive columns (if they exist)
candidates_df = candidates_df.drop(columns=["Gender", "Age"])
