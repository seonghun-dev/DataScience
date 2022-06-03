import pymysql as db

mydb = db.connect(host="localhost", user="root", password="6674", db='dataSceince')
cursor = mydb.cursor(db.cursors.DictCursor)

get_table = 'select sno, midterm, final from score_table where midterm >= 20 and final >=20 order by sno ASC'
cursor.execute(get_table)

row = cursor.fetchone()
while row:
    print(row['sno'], row['midterm'], row['final'])
    row = cursor.fetchone()
