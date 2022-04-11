import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from mlxtend.preprocessing import TransactionEncoder

data_df = pd.read_csv('data_week6.txt', header=None)
records = []
for i in range(len(data_df)):
    records.append(
        [str(data_df.values[i, j]) for j in range(len(data_df.columns)) if not pd.isna(data_df.values[i, j])])

te = TransactionEncoder()
te_ary = te.fit(records).transform(records, sparse=True)
te_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

frequent_itemset = apriori(te_df, min_support=0.005,use_colnames=True)
association_rules_df = association_rules(frequent_itemset, metric='confidence', min_threshold=0.005)
association_rules_conf_df = association_rules_df[association_rules_df['confidence'] >= 1]
print("d")
