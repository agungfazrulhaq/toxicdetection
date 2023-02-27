from kubernetes import client 
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1SKLearnSpec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import requests
import json
from minio import Minio
import pickle
import re
import streamlit as st

def remove_stopwords(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (string): sentence to remove the stopwords from

    Returns:
        sentence (string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    # Sentence converted to lowercase-only
    sentence = sentence.lower()

    words = sentence.split()
    no_words = [w for w in words if w not in stopwords]
    sentence = " ".join(no_words)

    return sentence
    
def remove_symbols(sentence) :
    return re.sub(r'[^\w]', ' ', sentence)


minio_client = Minio(
        "192.168.1.10:30950",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
minio_bucket = "mlpipeline"

EMBEDDING_DIM = 16
PADDING = 'post'
TRUNCATING = 'post'
MAXLEN = 150
    
minio_client.fget_object(minio_bucket, "commentoxic/tokenizer.pickle", "/tmp/tokenizer.pickle")
file = open('/tmp/tokenizer.pickle', 'rb')
tokenizer = pickle.load(file)

text = "you dont deserve to live cause you are a mexican and i hope you dead you nigger mexican"
text_after_stopwords = remove_stopwords(text)
text_clean = remove_symbols(text_after_stopwords)

text_sequence = tokenizer.texts_to_sequences([text_clean])
text_padded = pad_sequences(text_sequence, maxlen=MAXLEN, padding=PADDING, truncating=TRUNCATING)

labels = ['toxic','severe toxic','obscene','a threat','an insult','identity hate']
KServe = KServeClient()

isvc_url = "http://toxic-detection.researchai.svc.cluster.local/v1/models/toxic-detection:predict"

t = np.array(text_padded)
print(t.shape)
# t = t.reshape(-1,28,28,1)

inference_input = {
  'instances': t.tolist()
}

response = requests.post(isvc_url, json=inference_input)
r = json.loads(response.text)
predicted = r['predictions'][0]
# predicted = model.predict(text_padded)[0]

st.title('Toxic Comment Detection')

txt = st.text_area('Text to analyze', '''
    It was the best of times, it was the worst of times, it was
    the age of wisdom, it was the age of foolishness, it was
    the epoch of belief, it was the epoch of incredulity, it
    was the season of Light, it was the season of Darkness, it
    was the spring of hope, it was the winter of despair, (...)
    ''')
if st.button('Analyze'):
    st.write('Why hello there')

KServe = KServeClient()

iter_ = 0
for lab in labels :
    if predicted[iter_] > 0.5 :
        st.write("comment is "+lab+" ("+str(predicted[iter_]*100)+"%)")
    iter_ += 1