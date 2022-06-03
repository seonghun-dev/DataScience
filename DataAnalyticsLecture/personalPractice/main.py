import pandas as pd
from matplotlib import pyplot as plt
from nltk import word_tokenize, FreqDist
from nltk.tag import pos_tag
from wordcloud import WordCloud

from database import LocalDB

stopwords = ['semiconductor']

db = LocalDB()
abstracts_list = list(db.abstract.find({"$text": {"$search": "semiconductor"}}, {"_id": 0, "Id": 1}))
patent_list = list(db.patent.find({"$text": {"$search": "semiconductor"}}, {"_id": 0, "Id": 1}))
abstracts_list = [i["Id"] for i in abstracts_list]
patent_list = [i["Id"] for i in patent_list]
interested_patent_list = abstracts_list + patent_list
interested_patent_list = list(set(interested_patent_list))

abstracts_df = pd.DataFrame(list(db.abstract.find({"Id": {"$in": interested_patent_list}}, {"_id": 0})))
patents_df = pd.DataFrame(list(db.patent.find({"Id": {"$in": interested_patent_list}}, {"_id": 0})))
all_patents_df = pd.merge(left=abstracts_df, right=patents_df, how="inner", on="Id")
all_patents_df['firstCreatedAt'] = all_patents_df['firstCreatedAt'].apply(lambda x: int(x[0:4]))

year_patents_group = all_patents_df.groupby('firstCreatedAt').count()
year_patents_group_graph = year_patents_group['Id']
year_patents_group_graph = year_patents_group_graph.drop(labels=[2017])

year_patents_group_graph.plot()
plt.title("Patent Count by Year")
plt.xlabel("Year")
plt.ylabel("Number of patents")
plt.show()

range1 = range(2003, 2005)
range2 = range(2006, 2008)
range3 = range(2009, 2011)
range4 = range(2012, 2014)
range5 = range(2015, 2016)


def get_range_wc(df):
    df['abstract_noun'] = df['abstract'].apply(
        lambda x: [t[0] for t in pos_tag(word_tokenize(x)) if t[1] == "NN" and t[0] not in stopwords])
    noun_list = sum(df['abstract_noun'].tolist(), [])
    fd_names = FreqDist(noun_list)
    wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
    plt.imshow(wc.generate_from_frequencies(fd_names))
    plt.axis("off")
    plt.show()


df1 = all_patents_df[all_patents_df['firstCreatedAt'].isin(range1)]
r1 = get_range_wc(df1)

df2 = all_patents_df[all_patents_df['firstCreatedAt'].isin(range2)]
get_range_wc(df2)

df3 = all_patents_df[all_patents_df['firstCreatedAt'].isin(range3)]
get_range_wc(df3)

df4 = all_patents_df[all_patents_df['firstCreatedAt'].isin(range4)]
get_range_wc(df4)

df5 = all_patents_df[all_patents_df['firstCreatedAt'].isin(range5)]
get_range_wc(df5)
