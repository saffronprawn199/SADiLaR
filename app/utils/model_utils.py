# File: model_utils.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
from io import StringIO


# Create a logging-like interface that writes to a buffer
class StreamlitLogger(object):
    def __init__(self):
        self.buffer = StringIO()

    def write(self, message):
        # When a logging message is written, append it to the buffer
        self.buffer.write(message)
        # Then, use st.text to display the buffer in the app
        st.text(self.buffer.getvalue())

    def flush(self):
        # This flush method is required for compatibility with the logging module
        self.buffer.flush()


def cluster_embeddings(embeddings, threshold):
    clustering_model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold
    )
    clustering_model.fit(embeddings)
    return clustering_model.labels_


def create_tree_chart(df):
    fig = px.treemap(df, path=["Level 1", "Level 2"], values="Count")
    return fig


@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings
