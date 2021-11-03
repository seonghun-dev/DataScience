import time

import numpy as np
import pymysql
import statsmodels.api as sm


def load_dbscore_data():
    conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    sql = "select * from score"
    curs.execute(sql)

    data = curs.fetchall()
    curs.close()
    conn.close()
    x = [(t['attendance'], t['homework'], t['final']) for t in data]
    x = np.array(x)

    y = [(t['score']) for t in data]
    y = np.array(y)

    return x, y


X, y = load_dbscore_data()

X_const = sm.add_constant(X)

model = sm.OLS(y, X_const)
ls = model.fit()

print(ls.summary())

ls_c = ls.params[0]
ls_m1 = ls.params[1]
ls_m2 = ls.params[2]
ls_m3 = ls.params[3]


def gradient_descent_naive(x, y):
    epochs = 1000000
    min_grad = 0.0001
    learning_rate = 0.001

    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    c = 0.0

    n = len(y)

    c_grad = 0.0
    m1_grad = 0.0
    m2_grad = 0.0
    m3_grad = 0.0

    for epoch in range(epochs):
        for i in range(n):
            y_pred = m1 * x1[i] + m2 * x2[i] + m3 * x3[i] + c
            m1_grad += 2 * (y_pred - y[i]) * x1[i]
            m2_grad += 2 * (y_pred - y[i]) * x2[i]
            m3_grad += 2 * (y_pred - y[i]) * x3[i]
            c_grad += 2 * (y_pred - y[i])
        c_grad /= n
        m1_grad /= n
        m2_grad /= n
        m3_grad /= n

        m1 = m1 - learning_rate * m1_grad
        m2 = m2 - learning_rate * m2_grad
        m3 = m3 - learning_rate * m3_grad
        c = c - learning_rate * c_grad

        if epoch % 1000 == 0:
            print("epoch : %d m1_grad = %f m2_grad = %f m3_grad = %f c_grad=%f m1=%f m2=%f m3=%f c=%f" % (
                epoch, m1_grad, m2_grad, m3_grad, c_grad, m1, m2, m3, c))
        if abs(m1_grad) < min_grad and abs(m2_grad) < min_grad and abs(m3_grad) < min_grad and abs(
                c_grad) < min_grad:
            break
    return m1, m2, m3, c


start_time = time.time()
# m1, m2, m3, c = gradient_descent_naive(X, y)
end_time = time.time()

print("%f seconds" % (end_time - start_time))

print("\n\n final:")
# print("gdn_m1 = %f,gdn_m2 = %f,gdn_m3 = %f gdn_c=%f" % (m1, m2, m3, c))
print("ls_m1 = %f,ls_m2 = %f,ls_m3 = %f ls_c=%f" % (ls_m1, ls_m2, ls_m3, ls_c))


def gradient_descent_vectorized(x, y):
    epochs = 1000000
    min_grad = 0.0001
    learning_rate = 0.001

    m = [0.0, 0.0, 0.0]
    m = np.array(m)
    c = 0.0

    n = len(y)

    c_grad = 0.0
    m_grad = [0.0, 0.0, 0.0]

    for epoch in range(epochs):

        y_pred = np.dot(x, m) + c
        m_grad = (2 * np.dot((y_pred - y), x)) / n
        c_grad = (2 * (y_pred - y)).sum() / n
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if (epoch % 1000 == 0):
            print("epoch : %d m1_grad = %f m2_grad = %f m3_grad = %f c_grad=%f m1=%f m2=%f m3=%f c=%f" % (
                epoch, m_grad[0], m_grad[1], m_grad[0], c_grad, m[0], m[1], m[2], c))

        if abs(m_grad[0]) < min_grad and abs(m_grad[1]) < min_grad and abs(m_grad[2]) < min_grad and abs(
                c_grad) < min_grad:
            break

    return m[0], m[1], m[2], c


start_time = time.time()
m1, m2, m3, c = gradient_descent_vectorized(X, y)
end_time = time.time()

print("%f seconds" % (end_time - start_time))

print("\n\nFinal:")
print("gdv_m1 = %f,gdv_m2 = %f,gdv_m3 = %f gdv_c=%f" % (m1, m2, m3, c))
print("ls_m1 = %f,ls_m2 = %f,ls_m3 = %f ls_c=%f" % (ls_m1, ls_m2, ls_m3, ls_c))
