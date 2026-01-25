import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to recommend books and music
def recommendations(user_id, num_recom=7, cat="Books"):
    if cat == "Books":
        if user_id not in user_b_r.index:
            print(f"User ID {user_id} not found in the dataset.")
            return []
        
        user_ratings = user_b_r.loc[user_id]
        user_s_df = user_s_b_df
    elif cat == "Music":
        if user_id not in user_music_r.index:
            print(f"User ID {user_id} not found in the dataset.")
            return []
        
        user_ratings = user_music_r.loc[user_id]
        user_s_df = user_s_music_df
    else:
        return []

    common_books = user_ratings.index.intersection(user_s_df.columns)
    w_ratings = user_s_df[common_books].loc[user_id].values * user_ratings[common_books].values
    
    # Handle zero division and replace NaN with zero
    total_user_s_ratings = user_s_df[common_books].loc[user_id].sum()
    recomm_score = 0.0 if total_user_s_ratings == 0 else w_ratings.sum() / total_user_s_ratings
    
    unrated_items = user_ratings[user_ratings == 0].index
    recomm = recomm_score * (user_ratings == 0).astype(int)
    recomm = recomm[unrated_items].sort_values(ascending=False).head(num_recom)
    return recomm.index

# For Books
b = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Recommendation System\\Books.csv")
b_r = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Recommendation System\\Ratings.csv")
br = pd.merge(b, b_r, on="bookID")
br['Book-Rating'] = pd.to_numeric(br['Book-Rating'], errors='coerce')
br = br[br['Book-Rating'].notna()]
user_b_r = br.pivot_table(index="User-ID", columns="Book-Title", values="Book-Rating", aggfunc='mean')
user_b_r = user_b_r.fillna(0)

user_s_b = cosine_similarity(user_b_r)
user_s_b_df = pd.DataFrame(user_s_b, index=user_b_r.index, columns=user_b_r.index)

# For Music
music = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Recommendation System\\Music.csv")
ratings = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Recommendation System\\Music_Ratings.csv")
music_r = pd.merge(music, ratings, on="Title")
user_music_r = music_r.pivot_table(index="UserID", columns="Title", values="Ratings")
user_music_r = user_music_r.fillna(0)

user_s_music = cosine_similarity(user_music_r)
user_s_music_df = pd.DataFrame(user_s_music, index=user_music_r.index, columns=user_music_r.index)

# Get recommendations for Books
user_id_books = 23  # Example user_id
recomm_books = recommendations(user_id_books, cat="Books")
print(f"Recommended Books for User {user_id_books}:", recomm_books)

# Get recommendations for Music
user_id_music = 3  # Example user_id
recomm_music = recommendations(user_id_music, cat="Music")
print(f"Recommended Music for User {user_id_music}:", recomm_music)