import streamlit as st
st.title('Sentiment Analysis')

import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Ghannesh27/dataset/main/imdb.csv')
    
x=data["text"]
y=data["label"]
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
stop=stopwords.words("english")
stop.remove("not")
def clean_text(text):
    text = text.lower().strip()
    text = ' '.join(e for e in text.split() if e not in stop)
    return text
x=x.apply(clean_text)    
from nltk.stem.wordnet import WordNetLemmatizer
w = WordNetLemmatizer()
x=x.apply(lambda x:' '.join([w.lemmatize(word,'v') for word in x.split()])) 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())]) 

text_model.fit(x,y)

select = st.text_input('Enter your message')
op = text_model.predict([select])
if op[0]==0:
  a="Negative"
else:
  a="Positive "  
st.title(a)
