import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pymysql


def load_dbscore_data():
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


def main():
    X, y = load_dbscore_data()
    model = LinearRegression()
    model.train(X, y)
    model.animate(X, y)


class LinearRegression():
    def __init__(self, learning_rate=0.001, epochs=100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_grad = 0.0001
        self.c = 0
        self.m = 0
        self.w_list = []

    def train(self, X, y):
        n = len(y)
        c_grad = 0.0
        m_grad = 0.0

        for epoch in range(self.epochs):
            self.w_list.append([self.c, self.m])
            y_pred = self.m * X + self.c
            m_grad = (2 * (y_pred - y) * X).sum() / n
            c_grad = (2 * (y_pred - y)).sum() / n

            self.m = self.m - self.learning_rate * m_grad
            self.c = self.c - self.learning_rate * c_grad
            if abs(m_grad) < self.min_grad and abs(c_grad) < self.min_grad:
                break
        self.w_list = np.array(self.w_list)

    def animate(self, X, y):
        fig, ax = plt.subplots()
        ax.scatter(X, y)
        plot_range = np.array(range(int(min(X)) - 1, int(max(X)) + 3))
        a_0, a_1 = self.w_list[0,]
        y_plot = plot_range * a_1 + a_0
        ln, = ax.plot(plot_range, y_plot, color="red", label="Best Fit")

        def animator(frame):
            a_0, a_1 = self.w_list[frame,]
            y_plot = plot_range * a_1 + a_0
            ln.set_data(plot_range, y_plot)

        anim = animation.FuncAnimation(fig, func=animator, frames=self.epochs)
        anim.show()


if __name__ == "__main__":
    main()
