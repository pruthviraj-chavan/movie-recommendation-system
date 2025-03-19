import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned data
data = pd.read_csv('../data/cleaned_data.csv')

# Create a pivot table for user-item interactions (rows: users, columns: movies, values: ratings)
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute cosine similarity between movies
cosine_sim = cosine_similarity(user_movie_matrix.T)
similarity_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Function to get recommendations
def get_recommendations(movie_title, similarity_df, top_n=5):
    try:
        # Get similarity scores for the given movie
        scores = similarity_df[movie_title]
        
        # Sort movies by similarity score
        recommended_movies = scores.sort_values(ascending=False).index[1:top_n+1]  # Exclude the input movie itself
        
        return list(recommended_movies)
    except KeyError:
        return ["Movie not found in the dataset."]

# Example usage
if __name__ == '__main__':
    print(get_recommendations("Toy Story (1995)", similarity_df))