from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Import recommendation function
from models.collaborative_filtering_model import get_recommendations

app = Flask(__name__)

# Load cleaned data using an absolute path
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cleaned_data.csv')
data = pd.read_csv(data_path)

# Create user-movie matrix and similarity matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)
cosine_sim = cosine_similarity(user_movie_matrix.T)
similarity_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Route to serve the HTML form
@app.route('/')
def home():
    return '''
    <h1>Movie Recommendation System</h1>
    <form action="/recommend" method="get">
        <label for="movie">Enter a Movie Name:</label><br>
        <input type="text" id="movie" name="movie" required><br><br>
        <input type="submit" value="Get Recommendations">
    </form>
    '''

# Route to handle recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    # Get movie title from query parameter
    movie_title = request.args.get('movie')
    
    if not movie_title:
        return jsonify({"error": "Please provide a movie title."}), 400
    
    # Call recommendation function
    recommendations = get_recommendations(movie_title, similarity_df)
    
    return jsonify({"movie": movie_title, "recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)