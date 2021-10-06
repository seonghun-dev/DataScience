import numpy as np
import pymysql

conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from iris"
curs.execute(sql)

data = curs.fetchall()

curs.close()
conn.close()

x = [(t['sepal.length'], t['sepal.width'], t['petal.length'], t['petal.width']) for t in data]
x = np.array(x)

y = [1 if (t['variety'] == 'vesicolor') else -1 for t in data]
y = np.array(y)
print(y.shape)

# train data, test data 분류 작업
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn import tree

dtree = tree.DecisionTreeClassifier()

dtree.fit(x_train, y_train)
