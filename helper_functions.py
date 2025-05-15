import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
import os
from sklearn.decomposition import PCA

# All pages
def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    dataset_filename = './datasets/MelissaPathil_pMHC_TCR_dataset.csv'
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        if os.path.exists(dataset_filename):
            st.write("Loading dataset file: {}".format(dataset_filename))
            df = pd.read_csv(dataset_filename)
        else:
            st.write("File does not exist: {}".format(dataset_filename))
    if df is not None:
        st.session_state['data'] = df
    return df

# Page: Classify
def reduce_feature_dimensionality(X_train, X_test):
    """
        Reduce dimensions of training and test dataset features X_train_sentiment, X_val_sentiment
        Inputs:
            - X_train (pandas dataframe): training dataset features 
            - X_val_sentiment (pandas dataframe): test dataset features 
        Outputs:
            - X_train_sentiment (numpy array): training dataset features with smaller dimensions (number of columns)
            - X_val_sentiment (numpy array): test dataset features with smaller dimensions (number of columns)
        """
    # User input target number of output features
    num_comps = 6
    # Use PCA to reduce feature dimensions of training and test dataset features
    if len(X_train) > 0:
        pca = PCA(n_components=num_comps)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
    if len(X_test) > 0:
        pca.fit(X_test)
        X_test = pca.transform(X_test)
    return X_train, X_test