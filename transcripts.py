import json
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load JSON data from file ---
def load_transcripts(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# --- Text preprocessing ---
def clean_text(text):
    return text.lower().strip()

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Extract topics using OpenAI ---
def extract_topics(text):
    prompt = f"Extract key career-related themes or topics from this text:\n\n{text}\n\nReturn the topics as a comma-separated list."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()


# --- Normalize themes ---
def normalize_theme(theme):
    theme = theme.lower().strip()

    normalization_dict = {
        'software development': ['software dev', 'software engineering', 'programming', 'developer'],
        'data science': ['data analyst', 'data analytics', 'machine learning', 'ai', 'ml', 'artificial intelligence'],
        'medicine': ['doctor', 'healthcare', 'medical'],
        'law': ['lawyer', 'legal'],
        'engineering': ['civil engineering', 'mechanical engineering', 'engineer'],
        'business': ['entrepreneurship', 'business', 'startup', 'management'],
        'education': ['teacher', 'teaching', 'education'],
        'creative arts': ['music', 'artist', 'actor', 'acting', 'creative'],
    }

    for key, aliases in normalization_dict.items():
        if theme in aliases or key == theme:
            return key
    return theme  # If no match, return as-is

def normalize_topic_list(topics_str):
    raw_topics = [topic.strip() for topic in topics_str.split(",") if topic.strip()]
    normalized = [normalize_theme(topic) for topic in raw_topics]
    return list(set(normalized))  # Remove duplicates

# --- Load and process transcripts ---
transcripts = load_transcripts('transcripts.json')
transcripts['Cleaned'] = transcripts['content'].apply(clean_text)
transcripts['Chunks'] = transcripts['Cleaned'].apply(chunk_text)

# --- Extract and normalize topics for each chunk ---
def extract_all_topics(chunks):
    all_topics = []
    for chunk in chunks:
        raw = extract_topics(chunk)
        normalized = normalize_topic_list(raw)
        all_topics.extend(normalized)
    return list(set(all_topics))

transcripts['Topics'] = transcripts['Chunks'].apply(extract_all_topics)

# --- Print extracted themes ---
for i, row in transcripts.iterrows():
    print(f"\nTranscript {i+1} Themes:")
    for topic in row['Topics']:
        print("-", topic)

# --- Generate Word Cloud ---
all_topics = ', '.join([topic for topics in transcripts['Topics'] for topic in topics])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_topics)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Normalized Career Themes Word Cloud")
plt.show()
