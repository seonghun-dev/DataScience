import pandas as pd

document1 = '경찰청 철창살은 외철창살이냐 쌍철창살이냐 경찰청 철창살이 쇠철창살이냐 철철창살이냐검찰청 쇠철창살은 새쇠철창살이냐 헌쇠철창살이냐 경찰청 쇠창살 외철창살, 검찰청 쇠창살 쌍철창살.'

document2 = '내가 그린 기린 그림은 잘 그린 기린 그림이고 네가 그린 기린 그림은 잘 못 그린 기린 그림이다. 내가 그린 기린 그림은 긴 기린 그림이냐, 그냥 그린 기린 그림이냐? 내가 그린 구름그림은 새털구름 그린 구름그림이고, 네가 그린 구름그림은 깃털구름 그린 구름그림이다.'

document3 = '안촉촉한 초코칩 나라에 살던 안촉촉한 초코칩이 촉촉한 초코칩 나라의 촉촉한 초코칩을 보고 촉촉한 초코칩이 되고 싶어서 촉촉한 초코칩 나라에 갔는데 촉촉한 초코칩 나라의 촉촉한 초코칩 문지기가 "넌 촉촉한 초코칩이 아니고 안촉촉한 초코칩이니까 안촉촉한 초코칩나라에서 살아"라고해서 안촉촉한 초코칩은 촉촉한 초코칩이 되는것을 포기하고 안촉촉한 초코칩 나라로 돌아갔다.'


def replace_stopwords(tokens):
    stopwords = ['이니까', '이고', '이냐', '이다.', '이고', '에서', '이', '로', '은', '가', '에', '을', '의', ',', '.', ',',
                 '?', '"']
    for stopword in stopwords:
        tokens = [token.replace(stopword, '') for token in tokens]
    tokens = [token for token in tokens if token != '']
    return tokens


def replace_splitwords(document):
    splitwords = ['이냐']
    for splitword in splitwords:
        document = document.replace(splitword, ' ')
    document = document.replace('초코칩 나라', '초코칩나라')
    return document


def replace_parts(words_list):
    words_part = {'넌': '너', '네': '너', '내': '나', '그린': '그리다', '갔는데': '가다', '살던': '살다', '되고': '되다', '되는것을': '되다',
                  '살아라고해서': '살다', '돌아갔다': '돌아가다', '포기하고': '포기하다', '아니고': '아니다', '되는것': '되다', '싶어서': '싶다', '보고': '보다'}
    words_list = [words_part.get(word, word) for word in words_list]
    return words_list


preprocessing = lambda document: replace_parts(replace_stopwords(replace_splitwords(document).split()))


def tf_vector_space_model_boolean(document):
    """
    tf-based vector space model function boolean
    """
    # tokenize
    tokens = preprocessing(document)
    # count tokens
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1
    # calculate tf
    tf = {}
    for token in token_counts:
        tf[token] = 1
    return tf


def tf_vector_space_model_simple(document):
    """
    tf-based vector space model function simple
    """
    # tokenize
    tokens = preprocessing(document)
    # count tokens
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1
    return token_counts


def tf_vector_space_model_log(document):
    """
    tf-based vector space model function
    """
    # tokenize
    tokens = preprocessing(document)
    # count tokens
    token_counts = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1
    # calculate tf
    tf = {}
    for token in token_counts:
        tf[token] = token_counts[token] / len(tokens)
    return tf


bool_df = pd.DataFrame([tf_vector_space_model_boolean(document1), tf_vector_space_model_boolean(document2),
                        tf_vector_space_model_boolean(document3)],
                       index=['document1', 'document2', 'document3']).fillna(0).astype(int)
simple_df = pd.DataFrame([tf_vector_space_model_simple(document1), tf_vector_space_model_simple(document2),
                          tf_vector_space_model_simple(document3)],
                         index=['document1', 'document2', 'document3']).fillna(0).astype(int)
log_df = pd.DataFrame([tf_vector_space_model_log(document1), tf_vector_space_model_log(document2),
                       tf_vector_space_model_log(document3)],
                      index=['document1', 'document2', 'document3']).fillna(0).astype(float)

bool_df.to_csv('bool_df.csv')
simple_df.to_csv('simple_df.csv')
log_df.to_csv('log_df.csv')
