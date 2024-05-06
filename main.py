import pandas as pd
import numpy as np
import re
import string
import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
import Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from model import CustomKNNClassifier, CustomTfidfVectorizer

# Load saved model and vectorizer
knn_model = np.load('knn_model2.npy', allow_pickle=True).item()
tfidf_vectorizer = np.load('tfidf2.npy', allow_pickle=True).item()

def rm_special_char(comment):
  comment = comment.lower()
  comment = re.sub(r':', '', comment)
  comment = re.sub(r'[0-9]+', '', comment)
  comment = comment.translate(str.maketrans('', '', string.punctuation))
  return comment

def stopword_removal(comment):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(' '.join(comment)).split()

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return ' '.join([stemmer.stem(comment) for comment in text])



# Sample text preprocessing functions
def preprocess_tweet(tweet):
    tweet = ' '.join(word for word in tweet.split() if not word.startswith('@'))
    tweet = ' '.join(word for word in tweet.split() if not word.startswith('http'))
    tweet = rm_special_char(tweet)
    tweet = word_tokenize(tweet)
    tweet = stopword_removal(tweet)
    tweet = stemming(tweet)
    return tweet

# Load saved model and vectorizer


def testing(test_text):
    tfidf_test = tfidf_vectorizer.transform([test_text])
    predicted_sentiment = knn_model.predict(tfidf_test)
    if predicted_sentiment == 0:
        return "Predicted Sentiment: Positive"
    else:
        return "Predicted Sentiment: Negative"

def testing_files(test_text):
    tfidf_test = tfidf_vectorizer.transform([test_text])
    predicted_sentiment = knn_model.predict(tfidf_test)
    if predicted_sentiment == 0:
        return "Positive"
    else:
        return "Negative"

def main():
    st.title("Sentiment Analysis via Input Comment")
    text_input = st.text_input("Enter a tweet:")
    if st.button("Analyze"):
        processed_text = preprocess_tweet(text_input)
        result = testing(processed_text)
        st.write(result)
    
    st.title("Sentiment Analysis via Files")
    upl = st.file_uploader('Upload File', type=['csv'])
    column = st.text_input("Enter the comment column name")
    if st.button('Predict'):
        if upl is not None:
            file = pd.read_csv(upl)
            predictions = []
            kolom = []
            for comment in file[column]:
                processed_text = preprocess_tweet(comment)
                result = testing_files(processed_text)
                predictions.append(result)
                for numeric in file.dtypes:
                    if numeric == int:
                        kolom.append(numeric)
            file['Predictions'] = predictions
            sum = file['Predictions'].value_counts()
            st.write(file)
            st.write(sum)
            st.write(kolom)

if __name__ == "__main__":
    main()