#Script for Streamlit application using HuggingFace models
import pandas as pd
import streamlit as st
import transformers
import os

#text = transformers.__version__
#st.write("version", text)

#cwd_ = os.getcwd()
#st.write("Display", cwd_)

#env = os.environ
#st.write("env", env)
#os.environ["TRANSFORMERS_CACHE"] = cwd_

from transformers import pipeline
from transformers import AutoTokenizer
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource  # 👈 Add the caching decorator
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

@st.cache_resource  # 👈 Add the caching decorator
def load_tokenizer():
    return AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
tokenizer = load_tokenizer()
#cache_dir=cwd()
#st.write("directory", cache_dir)

@st.cache_resource  # 👈 Add the caching decorator
def load_sentiment_task():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer=tokenizer)
sentiment_task = load_sentiment_task()

@st.cache_data  # 👈 Add the caching decorator
def load_data(url):
    df = pd.read_csv(url)
    return df
dataset = load_data("metacritics_mario_odyssey.csv")
#first_record = dataset.iloc[0,:]
#first_comment = first_record['Comment']

comments = dataset['Comment'].astype('str')
comment_list = comments.values.tolist()

st.title("Social Media Comments")
st.subheader("Sentiment analysis and Summarization")

result = summarizer(comment_list, max_length=130, min_length=30, do_sample=False)
summary_table = pd.DataFrame(result)

sentiments = sentiment_task(comment_list)
sentiment_table = pd.DataFrame(sentiments)

sentiment_and_summary = pd.DataFrame(columns=['sentiment', 'summary'])
sentiment_and_summary['sentiment'] = sentiment_table['label']
sentiment_and_summary['summary'] = summary_table['summary_text']

#st.dataframe(summary_table)
#st.dataframe(sentiment_table)
st.dataframe(sentiment_and_summary)
#st.dataframe(comments)
#st.write("data type:",dtype)
#st.write('result:', result)

######################
#sidebar layout
######################

st.sidebar.title("Word Cloud")
# Create the word cloud image
word_cloud = WordCloud(background_color='white',
                       width=800,
                       height=400).generate(sentiment_and_summary.iloc[0,1])

# Display the generated image:
fig, ax = plt.subplots(figsize = (12, 8))
ax.imshow(word_cloud)
plt.axis("off")
st.sidebar.pyplot(fig)


# Sentiment Counts
st.sidebar.title("Sentiment Count")
sentiment_count = sentiment_and_summary.sentiment.value_counts().to_frame()
st.sidebar.bar_chart(data=sentiment_count)
#plt.xticks(rotation = 360)
#st.sidebar.pyplot(_)
