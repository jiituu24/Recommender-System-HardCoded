
import pandas as pd
import numpy as np

def item_based_recommendations(user_id, ratings, movies, top_n=5):

    movie_user = ratings.pivot(index='movieId', columns='userId', values='rating')
    similarity = movie_user.T.corr(method="pearson").fillna(0)

    user_ratings = ratings[ratings.userId == user_id]
    
    # Get movies already watched by user
    watched = set(user_ratings['movieId'].tolist())

    scores = {}

    for movie_id, rating in zip(user_ratings.movieId, user_ratings.rating):

        if movie_id not in similarity.columns:
            continue

        sim_movies = similarity[movie_id].drop(movie_id)

        for sim_movie, sim_score in sim_movies.items():
            if sim_movie not in watched:  # Only score unwatched movies
                scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating
    

    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rec_ids = [m for m, _ in sorted_movies[:top_n]]

    # titles = movies[movies.movieId.isin(rec_ids)]['title']
    # return titles.tolist()
    return rec_ids


def predict_item_rating(user_id, movie_id, ratings):

    movie_user = ratings.pivot(index='movieId', columns='userId', values='rating')

    if movie_id not in movie_user.index or user_id not in movie_user.columns:
        return ratings['rating'].mean()

    similarity = movie_user.T.corr(method="pearson").fillna(0)

    sim_movies = similarity[movie_id].drop(movie_id)

    user_ratings = movie_user[user_id].dropna()

    common_movies = sim_movies.index.intersection(user_ratings.index)

    if len(common_movies) == 0:
        return ratings['rating'].mean()

    sims = sim_movies[common_movies]
    vals = user_ratings[common_movies]

    denom = np.sum(np.abs(sims))

    if denom == 0:
        return ratings['rating'].mean()

    prediction = np.dot(sims, vals) / denom

    prediction = max(1, min(5, prediction))

    return float(prediction)
