import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    global word_list 
    word_list = []  
    lemmatizer = WordNetLemmatizer()  
    
    text = text.lower().strip() 
    words = word_tokenize(text)  

    processed_text = []  

    for word in words:
        if word.isalnum() and word not in string.punctuation and word not in stopwords.words("english"):
            lemmatized_word = lemmatizer.lemmatize(word)
            processed_text.append(lemmatized_word) 

    processed_text = ' '.join(processed_text)

    return processed_text

with open("./pklFiles/tfidfVectorize.pkl","rb") as f:
    tfidfVectorizer=pickle.load(f) 

with open("./pklFiles/pac.pkl","rb") as f:
    PAC_Model=pickle.load(f) 

st.header("Sentiment Analysis")
st1=st.text_area("Text To Analyze")
btn=st.button("Predict")
if btn:
    text=preprocess_text(st1)
    text_vec=tfidfVectorizer.transform([text])
    if PAC_Model.predict(text_vec)=="positive":
        st.subheader("Positive")
    elif PAC_Model.predict(text_vec)=="negative":
        st.subheader("Negative")
    elif PAC_Model.predict(text_vec)=="neutral":
        st.subheader("Neutral")
    else:
        st.subheader("Something Is Wrong")
