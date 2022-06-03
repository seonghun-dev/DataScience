# 건국대학교 스마트ICT융합공학과 201914308
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

xlfile = "./db_score.xlsx"

df = pd.read_excel(xlfile)
midterm = df['midterm']
final = df['final']
score = df['score']
grade = df['grade']

# (1) mean, median (midterm, final, score 각각)
print('(1) mean, median (midterm, final, score 각각)')
print("midterm mean =", midterm.mean(), "midterm median =", midterm.median())
print("final mean =", final.mean(), "final median =", final.median())
print("score mean =", score.mean(), "final median =", score.median())

# (2) mode (grade)
print('\n(2) mode (grade)')
print(grade.mode())

# (3) variance, standard deviation (midterm, final, score 각각)
print('\n(3) variance, standard deviation (midterm, final, score 각각)')
print("midterm variance =", np.var(midterm), "midterm standard deviation =", np.std(midterm))
print("final variance =", np.var(final), "final standard deviation =", np.std(final))
print("score variance =", np.var(score), "score standard deviation =", np.std(score))

# (4) percentile plot (midterm, final, score 각각)
plt.subplot(221)
plt.plot(midterm)
plt.title('midterm')

plt.subplot(222)
plt.plot(final)
plt.title('final')

plt.subplot(223)
plt.plot(score)
plt.title('score')

plt.suptitle('Percentile Plot')
plt.show()

# (5) boxplot (midterm, final, score 각각)
plt.subplot(221)
plt.boxplot(midterm)
plt.title('midterm')

plt.subplot(222)
plt.boxplot(final)
plt.title('final')

plt.subplot(223)
plt.boxplot(score)
plt.title('score')

plt.suptitle('boxplot')
plt.show()
# (6) histogram (midterm, final, score 각각)
plt.subplot(221)
plt.hist(midterm)
plt.title('midterm')

plt.subplot(222)
plt.hist(final)
plt.title('final')

plt.subplot(223)
plt.hist(score)
plt.title('score')

plt.suptitle('historgram')
plt.show()
# (7) scatter plot (midterm, final, score 에 대한, 모든 가능한 attribute 조합에 대하여 그릴 것) 4가지
plt.subplot(221)
plt.scatter(midterm, final)
plt.title('midterm and final')

plt.subplot(222)
plt.scatter(midterm, score)
plt.title('midterm and score')

plt.subplot(223)
plt.scatter(final, score)
plt.title('score and final')

plt.subplot(224)
plt.scatter(midterm, final, score)
plt.title('midterm and final and score')

plt.suptitle('scatter plot')
plt.show()
