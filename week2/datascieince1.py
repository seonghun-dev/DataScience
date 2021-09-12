import pandas as pd

filename = 'score.xlsx'
score = pd.read_excel(filename)
high_score = score[(score['midterm'] >= 20) & (score['final'] >= 20)]
print(high_score[['sno','midterm','final']].sort_values(by='sno' ,ascending=True))



