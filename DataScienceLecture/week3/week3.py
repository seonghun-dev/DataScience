import pandas as pd

xlfile = "./score.xlsx"

df = pd.read_excel(xlfile)
midterm = df['midterm']
final = df['final']

mean_midterm = midterm.mean()
mean_final = final.mean()

print("midterm  = ", mean_midterm)
print("final = ", mean_final)

cov = ((midterm - mean_midterm) * (final - mean_final)).mean()
print("cov = ", cov)

std_midterm = midterm.std()
std_final = final.std()
corr = cov / (std_midterm * std_final)  # correlation

print('correaltion = ', corr)
