import pandas as pd
import os
from user_based import user_based_recommendations, predict_user_rating
from item_based import item_based_recommendations, predict_item_rating
from evaluation import calculate_rmse
from precisionandrecall import precision_and_recall_at_k, evaluate_precision_recall

# Load datasets
movies = pd.read_csv("data/movies.csv")
train = pd.read_csv("data/ratings_train.csv")
test = pd.read_csv("data/ratings_test.csv")

print("Collaborative Filtering Recommender System")
print("------------------------------------------")

print("Training ratings:", len(train))
print("Testing ratings:", len(test))

actual = []
user_preds = []
item_preds = []

print("\nSample Predictions from Test Set")
print("User | Movie | Actual | UserCF | ItemCF")
print("------------------------------------------")

for i, row in enumerate(test.itertuples()):

    u = row.userId
    m = row.movieId
    r = row.rating

    user_pred = predict_user_rating(u, m, train)
    item_pred = predict_item_rating(u, m, train)

    actual.append(r)
    user_preds.append(user_pred)
    item_preds.append(item_pred)

    if i < 20:
        print(u, "|", m, "|", r, "|", round(user_pred,2), "|", round(item_pred,2))

user_rmse = calculate_rmse(actual, user_preds)
item_rmse = calculate_rmse(actual, item_preds)

print("\nAccuracy (RMSE)")
print("User Based CF:", round(user_rmse,3))
print("Item Based CF:", round(item_rmse,3))

print("\n--- Recommendation Demo ---")
user_id = int(input("Enter user id (1-100): "))

print("\nUser Based Recommendations:")
user_recs = user_based_recommendations(user_id, train, movies)

for m in user_recs:
    print(m)

print("\nItem Based Recommendations:")
item_recs = item_based_recommendations(user_id, train, movies)

for m in item_recs:
    print(m)


data = []

for m in user_recs:
    data.append(["User-Based", user_id, m])

for m in item_recs:
    data.append(["Item-Based", user_id, m])

df = pd.DataFrame(data, columns=["method","userId","movie"])

# Check if file exists to determine if we should write header
file_exists = os.path.exists("recommendations_output.csv")
df.to_csv("recommendations_output.csv", mode='a', header=not file_exists, index=False)

print("\nRecommendations saved to recommendations_output.csv")

# evaluate_precision_recall(train, test, movies, k=50)