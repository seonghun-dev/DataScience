import numpy as np
import pandas as pd
import pydot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df = pd.read_csv('data_week13.csv', index_col=0)
df['Species'] = df['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
X, y = np.array(df[df.columns.difference(['Species'])]), np.array(df['Species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dtree = DecisionTreeClassifier(max_depth=5)
dtree_model = dtree.fit(X_train, y_train)
y_pred = dtree_model.predict(X_test)
print(acc := accuracy_score(y_test, y_pred))

export_graphviz(dtree_model, out_file='dtree.dot', feature_names=df.columns.difference(['Species']),
                class_names=['setosa', 'versicolor', 'virginica'], rounded=True, filled=True)
(graph,) = pydot.graph_from_dot_file('./dtree.dot')
graph.write_png('dtree.png')
