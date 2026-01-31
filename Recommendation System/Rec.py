import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def build_user_item_matrix(df, user_col, item_col, rating_col):
    matrix = df.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col,
        aggfunc="mean"
    ).fillna(0)
    return matrix
def compute_user_similarity(user_item_matrix):
    similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
def recommend_items(
    user_id,
    user_item_matrix,
    user_similarity_df,
    top_k_users=5,
    top_n_items=7
):
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User ID {user_id} not found in dataset.")
    similar_users = (
        user_similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
        .head(top_k_users)
    )
    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    scores = {}

    for item in unrated_items:
        weighted_sum = 0
        sim_sum = 0

        for sim_user, sim_score in similar_users.items():
            rating = user_item_matrix.loc[sim_user, item]
            if rating > 0:
                weighted_sum += sim_score * rating
                sim_sum += sim_score

        scores[item] = 0 if sim_sum == 0 else weighted_sum / sim_sum
    recommendations = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(top_n_items)
    )

    return recommendations
books = pd.read_csv("C:/Users/Lenovo/Desktop/Recommendation System/Books.csv")
book_ratings = pd.read_csv("C:/Users/Lenovo/Desktop/Recommendation System/Ratings.csv")

book_data = pd.merge(books, book_ratings, on="bookID")
book_data["Book-Rating"] = pd.to_numeric(book_data["Book-Rating"], errors="coerce")
book_data.dropna(inplace=True)

user_book_matrix = build_user_item_matrix(
    book_data,
    user_col="User-ID",
    item_col="Book-Title",
    rating_col="Book-Rating"
)

book_user_similarity = compute_user_similarity(user_book_matrix)
music = pd.read_csv("C:/Users/Lenovo/Desktop/Recommendation System/Music.csv")
music_ratings = pd.read_csv("C:/Users/Lenovo/Desktop/Recommendation System/Music_Ratings.csv")

music_data = pd.merge(music, music_ratings, on="Title")

user_music_matrix = build_user_item_matrix(
    music_data,
    user_col="UserID",
    item_col="Title",
    rating_col="Ratings"
)

music_user_similarity = compute_user_similarity(user_music_matrix)
user_id_books = 23
book_recs = recommend_items(
    user_id=user_id_books,
    user_item_matrix=user_book_matrix,
    user_similarity_df=book_user_similarity
)

print(f"\nRecommended Books for User {user_id_books}:")
print(book_recs)
user_id_music = 3
music_recs = recommend_items(
    user_id=user_id_music,
    user_item_matrix=user_music_matrix,
    user_similarity_df=music_user_similarity
)

print(f"\nRecommended Music for User {user_id_music}:")
print(music_recs)
