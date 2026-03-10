import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import plotly.express as px

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/jobs.csv")

data = load_data()

st.set_page_config(page_title="CareerSense - AI Job Recommender", layout="wide")
st.title("💼 CareerSense: Context-Aware Job & Skill Recommendation System")

st.write("Upload your resume or enter your skills to get AI-powered job recommendations!")

# User input
user_input = st.text_area("✍️ Paste your resume text or list your skills:")

if st.button("Find Matching Jobs"):
    if user_input.strip() == "":
        st.warning("Please enter your resume or skills!")
    else:
        # Extract keywords using spaCy
        doc = nlp(user_input)
        user_skills = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        user_skills_text = " ".join(user_skills)

        # Combine job descriptions and skills
        data["combined"] = data["skills"].astype(str) + " " + data["description"]

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([user_skills_text] + data["combined"].tolist())

        # Compute similarity
        similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        data["match_score"] = similarity_scores

        # Top 3 matches
        top_matches = data.sort_values(by="match_score", ascending=False).head(3)

        st.subheader("🎯 Top Job Recommendations:")
        for i, row in top_matches.iterrows():
            st.markdown(f"**{row['job_title']}**")
            st.write(f"🧩 Skills: {row['skills']}")
            st.write(f"📄 Description: {row['description']}")
            st.progress(row["match_score"])
            st.write("---")

        # Visualization - Skill Match Scores
        fig = px.bar(top_matches, x="job_title", y="match_score", color="job_title",
                     title="Job Match Scores", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)


# Optional Skill Gap Visualization
if not user_input.strip() == "":
    matched_skills = []
    missing_skills = []
    for skill in data.iloc[data["match_score"].idxmax()]["skills"].split(","):
        if skill.strip().lower() in user_skills:
            matched_skills.append(skill.strip())
        else:
            missing_skills.append(skill.strip())

    st.subheader("📊 Skill Gap Analysis")
    skill_data = pd.DataFrame({
        "Skill Type": ["Matched"] * len(matched_skills) + ["Missing"] * len(missing_skills),
        "Skill": matched_skills + missing_skills
    })
    fig2 = px.bar(skill_data, x="Skill", color="Skill Type", title="Matched vs Missing Skills")
    st.plotly_chart(fig2, use_container_width=True)



