import pandas as pd

# Load raw datasets
movies = pd.read_csv('data/ml-latest-small/movies.csv')
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

# Display first few rows
print("Movies Dataset:")
print(movies.head())
print("\nRatings Dataset:")
print(ratings.head())

# Merge movies and ratings
data = pd.merge(ratings, movies, on='movieId')

# Drop unnecessary columns
data = data[['userId', 'movieId', 'rating', 'title', 'genres']]

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Save cleaned data
data.to_csv('data/cleaned_data.csv', index=False)
print("\nCleaned data saved to 'data/cleaned_data.csv'")