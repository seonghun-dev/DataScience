from pymongo import MongoClient


def LocalDB():
    client = MongoClient('localhost:27017')
    db = client.data
    return db
