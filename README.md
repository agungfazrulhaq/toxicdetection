# Toxic Comment classification

This repository contains the code for a machine learning project focused on identifying toxic comments in english. The project uses a dataset of comments labeled as 6 possible toxicity category (toxic, severe toxic, obscene, threat, insult, and identity hate) to train a classification model.

- '<b>toxic_classification_training.ipynb</b>': is a Jupyter notebook that contains the code for training the machine learning model. This notebook reads the dataset, preprocesses the text data, trains the model, and evaluates its performance.
- '<b>toxic_classification_training_pipeline.ipynb</b>': a Jupyter notebook that contains the code for building a kubeflow pipeline based on training Jupyter notebook.
- '<b>inference.ipynb</b>': model inference test with jupyter notebook using Kserve
- '<b>webapp</b>': directory contains web app with streamlit library to demonstrate model inference
- '<b>data</b>': contains data training and test
- '<b>configs</b>': contains config file (.yaml) to configure access to kubeflow pipeline and model serving

## References
- Data : https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
