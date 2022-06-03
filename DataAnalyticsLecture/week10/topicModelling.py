import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Load data
df = pd.read_csv('data_week10.txt', header=None)
data = sum(df.values.tolist(), [])

for topic_num in range(2, 6):
    count_vect = CountVectorizer(max_df=0.95, max_features=1000,
                                 min_df=2, stop_words='english',
                                 ngram_range=(1, 2))
    ftr_vect = count_vect.fit_transform(data)

    lda = LatentDirichletAllocation(n_components=topic_num, random_state=42)
    lda.fit(ftr_vect)

    doc_topics = lda.transform(ftr_vect)

    topic_names = ['Topic #' + str(i) for i in range(topic_num)]
    topic_df = pd.DataFrame(data=doc_topics, columns=topic_names, index=data)
    topic_df.to_csv('data_week10_topic_' + str(topic_num) + '.csv')
