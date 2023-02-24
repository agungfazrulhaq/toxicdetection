import streamlit as st
import numpy as np
from kubernetes import client 
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1SKLearnSpec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import requests
import json
from minio import Minio
import pickle
import re

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

isvc_resp = KServe.get("toxic-comment-2023-02-23--08-01-38", namespace="researchai")
isvc_url = isvc_resp['status']['address']['url']
st.write(isvc_url)