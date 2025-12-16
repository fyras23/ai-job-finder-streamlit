# app.py → FINAL DEPLOYABLE VERSION FOR STREAMLIT CLOUD
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="AI Job Finder", layout="wide")
st.title("🚀 AI-Powered Job Recommender")
st.markdown("### Search **123,849+ real job postings** with smart filters & beautiful insights")

# === LOAD DATA AND MODELS FROM HUGGING FACE ===
@st.cache_resource(show_spinner="Downloading 123k+ jobs & AI models from Hugging Face (first time only)...")
def load_model():
    # ⚠️ CHANGE THIS TO YOUR ACTUAL HUGGING FACE REPO
    REPO_ID = "fyras23/rec"  # Example: "firas/job-recommender-assets"

    # Download files (automatically cached after first download)
    tfidf_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
    matrix_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_matrix.pkl")
    df_path = hf_hub_download(repo_id=REPO_ID, filename="jobs_df.pkl")

    # Load them
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    
    with open(matrix_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    df = pd.read_pickle(df_path)

    # Clean state column for reliable filtering
    df['state'] = df['state'].fillna('').astype(str).str.upper().str.strip()

    return tfidf, tfidf_matrix, df

# Load everything
tfidf, tfidf_matrix, df = load_model()

# Prepare state dropdown
valid_states = [s for s in df['state'] if len(s) == 2]
all_states = ["Any State"] + sorted(set(valid_states)) + ["Remote Only"]

# === SIDEBAR FILTERS ===
st.sidebar.header("🔍 Search Filters")
query = st.sidebar.text_input("Job title or keywords", placeholder="e.g. nurse injector, data scientist, python developer")

col1, col2 = st.sidebar.columns(2)
min_salary = col1.number_input("Min Salary ($)", 0, 500000, 80000, 10000)
max_salary = col2.number_input("Max Salary ($)", 0, 1000000, 400000, 10000)

selected_state = st.sidebar.selectbox("Preferred State", all_states)
top_n = st.sidebar.slider("Number of results", 5, 30, 10)
salary_weight = st.sidebar.slider("Prioritize higher salary", 0.0, 1.0, 0.4, 0.05)

# === SEARCH BUTTON ===
if st.sidebar.button("Search Jobs", type="primary"):
    if not query.strip():
        st.warning("Please enter a job title or keywords")
        st.stop()

    with st.spinner("Searching through 123,849 jobs..."):
        # Clean query
        q_clean = re.sub(r'[^a-z0-9\s]', ' ', query.lower())
        q_vec = tfidf.transform([q_clean])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

        scores = sims.copy()

        # Salary filter
        salary_ok = (df['salary'] >= min_salary) & (df['salary'] <= max_salary)
        scores[~salary_ok] = 0

        # Salary boost
        if salary_weight > 0:
            sal_norm = df['salary'].fillna(df['salary'].median())
            sal_norm = (sal_norm - sal_norm.min()) / (sal_norm.max() - sal_norm.min() + 1e-8)
            scores = (1 - salary_weight) * scores + salary_weight * sal_norm

        # Location filter
        if selected_state != "Any State":
            if selected_state == "Remote Only":
                loc_ok = df['location'].astype(str).str.contains("remote|united states|anywhere", case=False, na=False)
            else:
                loc_ok = df['state'] == selected_state
            scores[~loc_ok] = 0

        # Get top results
        top_idx = np.argsort(scores)[::-1][:top_n]
        top_scores = scores[top_idx]

        if top_scores.max() == 0:
            st.error("No jobs found 😔 Try broadening your search or adjusting filters!")
            st.stop()

        results = df.iloc[top_idx].copy()
        results['score'] = top_scores

        # === DISPLAY RESULTS ===
        st.success(f"Found **{len(results)} top jobs** for **'{query}'** 🎉")

        # Average salary metric
        avg_salary = results['salary'].mean()
        if pd.notna(avg_salary):
            st.metric("Average Salary in Results", f"${avg_salary:,.0f}/year")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Job Titles")
            title_counts = results['title'].value_counts().head(8)
            fig1 = px.bar(
                y=title_counts.index,
                x=title_counts.values,
                orientation='h',
                color=title_counts.values,
                color_continuous_scale="Viridis",
                labels={"x": "Number of Jobs", "y": "Job Title"}
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("State Distribution")
            state_counts = results['state'].value_counts()
            if len(state_counts) > 1:
                fig2 = px.pie(
                    values=state_counts.values,
                    names=state_counts.index,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("All results from one state")

        # Job Listings
        st.subheader("📋 Recommended Job Listings")
        for i, (_, row) in enumerate(results.iterrows(), 1):
            salary_str = f"${row['salary']:,.0f}/yr" if pd.notna(row['salary']) else "Salary not disclosed"
            company = row['company_name'] if pd.notna(row['company_name']) else "Confidential"
            location = row['location'] if pd.notna(row['location']) else "Location not specified"

            with st.expander(f"{i}. **{row['title']}** • {company} • {salary_str} • {location}"):
                st.caption(f"Match Score: {row['score']:.3f} (higher = better match)")
                desc = str(row['description'])
                st.write(desc[:1500] + ("..." if len(desc) > 1500 else ""))

else:
    st.info("👈 Use the sidebar to search for jobs!")
    st.balloons()

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • TF-IDF + Cosine Similarity • Data from 123,849+ real job postings")