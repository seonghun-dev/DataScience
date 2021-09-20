import pandas as pd
import pymysql as db

filename = 'score.xlsx'
score = pd.read_excel(filename)

mydb = db.connect(host = "localhost",user = "root",password = "6674", db='dataSceince')
cursor = mydb.cursor()
make_table = 'create table if not exists score_table(sno int primary key, attendance float, homework float, discussion ' \
           'int, midterm float,final float,score float, grade char(1)) '
cursor.execute(make_table)

sql = 'insert into score_table(sno, attendance, homework, discussion, midterm, final, score, grade) values(%s, %s, %s, %s, %s, %s, %s, %s)'
for idx in range(len(score)):
	cursor.execute(sql, tuple(score.values[idx]))
mydb.commit()
