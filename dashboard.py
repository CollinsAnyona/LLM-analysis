import streamlit as st
import pandas as pd
import json
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime

# Optional for summarization
from transformers import pipeline

# Load summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Page config
st.set_page_config(
    page_title="Student Career Aspirations Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply improved CSS styles
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #FAFAFB;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #006391, #AE3E5B);
            padding: 1.5rem 1rem;
            color: white;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: white !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.5rem;
        }
        div[role="radiogroup"] > label[data-testid="stRadioButton"] {
            background-color: rgba(255, 255, 255, 0.15);
            border-radius: 5px;
            padding: 0.4rem 0.8rem;
        }
        div[role="radiogroup"] > label[data-testid="stRadioButton"]:hover {
            background-color: rgba(255, 255, 255, 0.25);
        }
        h1 { color: #1E1E2F; font-weight: 700; }
        h2 { color: #3B3B58; margin-top: 1rem; font-size: 1.6rem; }
        .stDataFrame, .stPlotlyChart, .stPyplot {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Sidebar nav
st.sidebar.title("Dashboard Navigation")
options = st.sidebar.radio("Select a Section", (
    "Overview", "Career Interests", "Topic Modeling", "Topic Distribution", "Sentiment Analysis", "Clustering", "Summary Insight"
))

st.title("Student Career Aspirations Dashboard")

# Load data
with open("mock_data.json", "r") as f:
    mock_data = json.load(f)
df = pd.DataFrame(mock_data)
text_data_for_modeling = df['career_goals'].str.lower()

# Location filter
if "location" in df.columns:
    locs = st.sidebar.multiselect("Filter by Location", options=df['location'].unique())
    if locs:
        df = df[df['location'].isin(locs)]

# Overview
if options == "Overview":
    st.subheader("Student Career Data Overview")
    st.dataframe(df)
    st.download_button("Download Data as CSV", df.to_csv(index=False), file_name="filtered_data.csv")

# Career Interests
elif options == "Career Interests":
    st.subheader("Career Interests")
    career_interest_count = df['interest_area'].value_counts()
    fig = px.bar(career_interest_count, x=career_interest_count.index, y=career_interest_count.values,
                 title="Career Interests", labels={'x': 'Career Area', 'y': 'Count'})
    st.plotly_chart(fig)
    st.subheader("Word Cloud")
    wc = WordCloud(width=600, height=400, background_color='white').generate(' '.join(career_interest_count.index))
    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Topic Modeling
elif options == "Topic Modeling":
    st.subheader("Topic Modeling")
    uploaded_file = st.file_uploader("Upload Transcript", type=["txt"])
    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        text_data_for_modeling = [raw_text.lower()]
        st.success("Uploaded successfully")
        st.text_area("Preview", raw_text[:500])

    n_topics = st.slider("Number of Topics", 2, 10, 5)
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(text_data_for_modeling)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()

    for i, topic in enumerate(lda.components_):
        word_freq = {words[j]: topic[j] for j in topic.argsort()[:-11:-1]}
        wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)
        st.subheader(f"Topic #{i+1}")
        plt.figure(figsize=(10,6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

# Topic Distribution
elif options == "Topic Distribution":
    st.subheader("Topic Distribution")
    topic_dist = lda.transform(X)
    df_topic_dist = pd.DataFrame(topic_dist, columns=[f"Topic {i+1}" for i in range(n_topics)])
    st.dataframe(df_topic_dist)
    topic_avg = topic_dist.mean(axis=0)
    fig = px.pie(values=topic_avg, names=[f"Topic {i+1}" for i in range(n_topics)], title="Topic Distribution")
    st.plotly_chart(fig)

# Sentiment Analysis
elif options == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")
    df['polarity'] = df['career_goals'].apply(lambda x: TextBlob(x).sentiment.polarity)
    fig = px.histogram(df, x='polarity', nbins=20, title="Sentiment Polarity Distribution")
    st.plotly_chart(fig)

# Clustering
elif options == "Clustering":
    st.subheader("Career Goal Clusters")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['career_goals'])
    k = st.slider("Select Number of Clusters", 2, 10, 4)
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(tfidf_matrix)
    df['cluster'] = km.labels_
    st.dataframe(df[['career_goals', 'cluster']])
    fig = px.histogram(df, x='cluster', title="Cluster Distribution")
    st.plotly_chart(fig)

# Summary Insight
elif options == "Summary Insight":
    st.subheader("Summary Insight")
    all_text = " ".join(df['career_goals'].tolist())
    if len(all_text) > 100:
        summary = summarizer(all_text[:1024])[0]['summary_text']
        st.markdown(f"**Generated Summary:** {summary}")
    else:
        st.info("Not enough text data to summarize.")

st.markdown("---")
st.markdown("Developed by Collins Otieno Jr.")
