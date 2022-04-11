import pandas as pd
import pymysql


def load_iris_data():
    csv_file = './iris.csv'
    iris = pd.read_csv(csv_file)

    conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    drop_sql = """drop table if exists iris"""
    curs.execute(drop_sql)
    conn.commit()

    import sqlalchemy
    database_username = "root"
    database_password = "6674"
    database_ip = "localhost"
    database_name = "dataSceince"
    database_connection = sqlalchemy.create_engine(
        'mysql+pymysql://{0}:{1}@{2}/{3}'.format(database_username, database_password, database_ip, database_name))
    iris.to_sql(con=database_connection, name="iris", if_exists='replace')


load_iris_data()
