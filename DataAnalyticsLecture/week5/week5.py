import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

user_id = 11
df = pd.read_csv('data_week5.txt')
user_x = pd.DataFrame(
    [[user_id, 2, 2], [user_id, 3, 1], [user_id, 5, 3], [user_id, 8, 4], [user_id, 9, 3]],
    columns=['User', 'Item', 'Score'])
df = df.append(user_x, ignore_index=True)

pt = df.pivot_table('Score', index='User', columns='Item')
pt.fillna(0, inplace=True)

user_based_collab = cosine_similarity(pt, pt)
user_based_collab = pd.DataFrame(user_based_collab, index=pt.index, columns=pt.index)


def predict_ratings(user, item):
    neighbors_ratings = pt[item].drop(index=user)
    neighbors_sim = user_based_collab[user].drop(index=user)
    del_idx = [idx + 1 for idx, val in enumerate(neighbors_ratings) if val == 0]
    for idx in del_idx:
        del neighbors_sim[idx]
        del neighbors_ratings[idx]

    score = (neighbors_sim * neighbors_ratings).sum() / neighbors_sim.sum()
    return score


item_list = [idx + 1 for idx, val in enumerate(pt.loc[user_id]) if val == 0]
for i in item_list:
    print(f'{i} - {predict_ratings(user_id, i)}')
