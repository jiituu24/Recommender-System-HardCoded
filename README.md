# Movie Recommendation System using Collaborative Filtering

## Project Overview
This project implements and compares two collaborative filtering approaches for generating movie recommendations: **User-Based Collaborative Filtering** and **Item-Based Collaborative Filtering**. The system analyzes user ratings to predict preferences and suggest relevant movies.

## Features

### Recommendation Algorithms
- **User-Based Collaborative Filtering**: Recommends movies by finding similar users and suggesting movies they rated highly
- **Item-Based Collaborative Filtering**: Recommends movies by analyzing similarity between movies based on user rating patterns

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy across test data
- **Precision & Recall**: Evaluates recommendation quality at different cutoff thresholds (e.g., top-5, top-10)

### Core Functions
- `user_based_recommendations()`: Generates recommendations using similar users
- `item_based_recommendations()`: Generates recommendations based on item similarity
- `predict_user_rating()` / `predict_item_rating()`: Predicts ratings for user-movie pairs
- `calculate_rmse()`: Evaluates prediction error
- `precision_and_recall_at_k()`: Measures recommendation precision and recall

## Dataset
The project uses a movie ratings dataset containing:
- `movies.csv`: Movie metadata (movieId, title)
- `ratings_train.csv`: Training set of user ratings
- `ratings_test.csv`: Test set for evaluation
- `ratings_full.csv`: Complete ratings dataset

## Project Structure
```
├── main.py                    # Main execution script
├── user_based.py             # User-based CF implementation
├── item_based.py             # Item-based CF implementation
├── evaluation.py             # RMSE calculation
├── precisionandrecall.py      # Precision/Recall metrics
├── recommendations_output.csv # Output recommendations
└── data/                      # Dataset directory
```

## How It Works
1. **Data Loading**: Loads movie and rating datasets
2. **Model Training**: Computes similarity matrices using Pearson correlation
3. **Prediction**: For each user-movie pair, predicts ratings using both methods
4. **Evaluation**: Compares RMSE scores and generates recommendation metrics
5. **Demo**: Provides sample recommendations for users

## Usage
Run the main script to execute the complete pipeline:
```bash
python main.py
```

This will output prediction comparisons, accuracy metrics (RMSE), and sample recommendations.

## Key Insights
The system allows comparison of two fundamental collaborative filtering approaches to determine which performs better on your dataset in terms of prediction accuracy and recommendation quality.
