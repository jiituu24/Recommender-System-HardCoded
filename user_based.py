
import pandas as pd
import numpy as np

def user_based_recommendations(user_id, ratings, movies, top_n=5):

    user_item = ratings.pivot(index='userId', columns='movieId', values='rating')
    similarity = user_item.T.corr(method="pearson").fillna(0)

    if user_id not in similarity.columns:
        return []

    sim_scores = similarity[user_id].drop(user_id)
    similar_users = sim_scores.sort_values(ascending=False).head(5).index

    watched = ratings[ratings.userId == user_id]['movieId'].tolist()

    rec_movies = ratings[
        (ratings.userId.isin(similar_users)) &
        (~ratings.movieId.isin(watched))
    ]

    movie_scores = rec_movies['movieId'].value_counts().head(top_n)

    # titles = movies[movies.movieId.isin(movie_scores.index)]['title']

    # return titles.tolist()
    return movie_scores.index.tolist()

def predict_user_rating(user_id, movie_id, ratings):

    user_item = ratings.pivot(index='userId', columns='movieId', values='rating')

    if movie_id not in user_item.columns or user_id not in user_item.index:
        return ratings['rating'].mean()

    similarity = user_item.T.corr(method="pearson").fillna(0)

    sim_users = similarity[user_id].drop(user_id)

    movie_ratings = user_item[movie_id].dropna()

    common_users = sim_users.index.intersection(movie_ratings.index)

    if len(common_users) == 0:
        return ratings['rating'].mean()

    sims = sim_users[common_users]
    vals = movie_ratings[common_users]

    user_means = user_item.loc[common_users].mean(axis=1)

    target_user_mean = user_item.loc[user_id].mean()

    denom = sum(abs(sims))

    if denom == 0:
        return ratings['rating'].mean()

    prediction = target_user_mean + sum(sims * (vals - user_means)) / denom

    prediction = max(1, min(5, prediction))

    return float(prediction)
