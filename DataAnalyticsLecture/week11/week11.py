import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

df = pd.read_csv('data_week11.csv')
data = df[['V1', 'V2']]

scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 10))
r = visualizer.fit(data_scale)
k = r.elbow_value_
plt.show()

model = KMeans(n_clusters=k, random_state=0)
model.fit(data_scale)
df['cluster'] = model.fit_predict(data_scale)

plt.figure(figsize=(10, 10))
for i in range(k):
    plt.scatter(data_scale[df['cluster'] == i, 0], data_scale[df['cluster'] == i, 1], label='cluster' + str(i))
plt.legend()
plt.show()
