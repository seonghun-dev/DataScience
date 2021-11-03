## 중간고사 이용해 최종 점수 예측
import time

import numpy as np
import pymysql
import statsmodels.api as sm
from matplotlib import pyplot as plt


def load_dbsocore_data():
    conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    sql = "select * from score"
    curs.execute(sql)

    data = curs.fetchall()
    curs.close()
    conn.close()

    x = [(t['midterm']) for t in data]
    x = np.array(x)

    y = [(t['score']) for t in data]
    y = np.array(y)

    return x, y


x, y = load_dbsocore_data()

x_const = sm.add_constant(x)

model = sm.OLS(y, x_const)
ls = model.fit()

print(ls.summary())

ls_c = ls.params[0]
ls_m = ls.params[1]

y_pred = ls_m * x + ls_c

plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')
plt.show()


def gradient_descent_naive(x, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001

    m = 0.0
    c = 0.0

    n = len(y)

    c_grad = 0.0
    m_grad = 0.0

    for epochs in range(epochs):
        for i in range(n):
            y_pred = m * x[i] + c  # 현재 i번째 샘플에 대한 예측값
            m_grad += 2 * (y_pred - y[i]) * x[i]
            c_grad = 2 * (y_pred - y[i])
        c_grad /= n
        m_grad /= n

        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if epochs % 1000 == 0:
            print("epoch : %d m_grad = %f c_grad=%f m=%f c=%f" % (epochs, m_grad, c_grad, m, c))
        if abs(m_grad) < min_grad and abs(c_grad) < min_grad:
            break
    return m, c


start_time = time.time()
m, c = gradient_descent_naive(x, y)
end_time = time.time()

print("%f seconds" % (end_time - start_time))

print("\n\n final:")
print("gd_m = %f, gd_c=%f" % (m, c))
print("ls_m = %f ls_c = %f" % (ls_m, ls_c))
