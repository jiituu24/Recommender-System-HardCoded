import pandas as pd
import numpy as np

from user_based import user_based_recommendations, predict_user_rating
from item_based import item_based_recommendations, predict_item_rating

def precision_and_recall_at_k(recommendation_ids, test_ratings, user_id, k = 5,threshold=3.5) :

    liked = set(
        test_ratings[(test_ratings.userId == user_id) & (test_ratings.rating >= threshold)]['movieId']
    )

    if(len(liked) == 0) :
        return (0, 0) 


    if(len(recommendation_ids) < k) :
        k = len(recommendation_ids)
        
    top_k = recommendation_ids[:k]
    hits = len(set(top_k) & liked)

    return (hits / k), (hits / len(liked))


def evaluate_precision_recall(train_ratings, test_ratings ,movies, k = 50) :

    for user_id in set(test_ratings.userId) :

        ub = user_based_recommendations(user_id, train_ratings, movies, k)
        tb = item_based_recommendations(user_id, train_ratings, movies, k)

        ub_prec, ub_rec = precision_and_recall_at_k(ub, test_ratings, user_id, k)
        tb_prec, tb_rec = precision_and_recall_at_k(tb, test_ratings, user_id, k)

        print(f"User {user_id} - UserCF: Precision@{k}={ub_prec:.2f}, Recall@{k}={ub_rec:.2f} | ItemCF: Precision@{k}={tb_prec:.2f}, Recall@{k}={tb_rec:.2f}")


if __name__ == "__main__":
    train_ratings = pd.read_csv("data/ratings_train.csv")
    test_ratings = pd.read_csv("data/ratings_test.csv")
    movies = pd.read_csv("data/movies.csv")
    
    evaluate_precision_recall(train_ratings, test_ratings, movies, k=10)

