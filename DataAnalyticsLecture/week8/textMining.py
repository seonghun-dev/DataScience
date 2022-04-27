import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize

from DataAnalyticsLecture.week8.constant import stopwords


def tf_vector_space_model_simple(tokens):
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1
    return token_counts


def get_network(df):
    plt.figure(figsize=(40, 40))
    g = nx.from_pandas_adjacency(df)
    nx.draw(g, with_labels=True, node_size=3000, font_size=9, font_color='#FFFFFF', edge_color='#a9a9a9')
    plt.show()
    return g


dataset = pd.read_json('data_week8.json')
check_stopwords = lambda words: [t.lower() for t in words if t.lower() not in stopwords and t.isalpha()]
review = dataset['review_body'].apply(lambda x: check_stopwords(word_tokenize(x))).tolist()
simple_df = pd.DataFrame([tf_vector_space_model_simple(document) for document in review]).fillna(0).astype(int)
sum_results = simple_df.sum()
del_col_idx = [idx for idx, val in enumerate(sum_results) if val <= 4]
simple_df = simple_df.drop(simple_df.columns[del_col_idx], axis=1)
term_correlation_df = simple_df.T.dot(simple_df)
np.fill_diagonal(term_correlation_df.values, 0)
get_network(term_correlation_df)
