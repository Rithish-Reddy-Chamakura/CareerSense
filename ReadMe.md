# 💼 CareerSense: AI powered job and skill recommendation system

## 📖 Overview
CareerSense is an **AI-powered career recommendation system** that matches users to the most relevant jobs based on their skills, resume content.  
It uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze user profiles and job data, providing personalized job matches.

---

## 🎯 Objectives
- Analyze resume text to extract key skills and domains.
- Match user profiles with the most relevant job descriptions.
- Visualize job match scores for better understanding.
- Suggest a career direction based on user strengths.

---

## ⚙️ Tech Stack
| Layer | Tools / Libraries |
|-------|--------------------|
| Data Processing | Pandas, NumPy |
| NLP | spaCy |
| Machine Learning | scikit-learn (TF-IDF, Cosine Similarity) |
| Visualization | Plotly |
| Web App | Streamlit |

---

## 🧩 How It Works
1. **User Input** – Upload or type resume text / skills.
2. **Skill Extraction** – Uses spaCy NLP to identify relevant skills.
3. **Vectorization** – TF-IDF converts skills and job data into feature vectors.
4. **Similarity Matching** – Cosine similarity ranks the most relevant jobs.
5. **Visualization** – Interactive bar chart of match scores using Plotly.

---

## 🧠 Dataset
Create a simple dataset named `jobs.csv` under `/data/`.

Example:
```csv
job_id,job_title,skills,description
1,Data Scientist,"Python, Machine Learning, SQL, Statistics","Analyze data and build ML models"
2,Web Developer,"HTML, CSS, JavaScript, React","Design and build modern web apps"
3,AI Engineer,"Deep Learning, TensorFlow, NLP, Python","Develop and deploy AI solutions"
4,Data Analyst,"Excel, Power BI, SQL, Data Visualization","Analyze business data and create dashboards"

