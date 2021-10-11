import pandas as pd
import pymysql


def load_db_score():
    excel_file = 'db_score_3_labels.xlsx'
    db_score_3 = pd.read_excel(excel_file)

    conn = pymysql.connect(host="localhost", user="root", password="6674", db='dataSceince')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    drop_sql = """drop table if exists db_score_3"""
    curs.execute(drop_sql)
    conn.commit()

    import sqlalchemy
    database_username = "root"
    database_password = "6674"
    database_ip = "localhost"
    database_name = "dataSceince"
    database_connection = sqlalchemy.create_engine(
        'mysql+pymysql://{0}:{1}@{2}/{3}'.format(database_username, database_password, database_ip, database_name))
    db_score_3.to_sql(con=database_connection, name="db_score_3", if_exists='replace')


load_db_score()
