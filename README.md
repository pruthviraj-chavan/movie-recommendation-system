We implemented Item-Based Collaborative Filtering by:

Preprocessing the MovieLens dataset to create a clean user-movie matrix.
Computing cosine similarity between movies to identify similar items.
Building a Flask API to allow users to input a movie name and receive personalized recommendations.

# Data processing and analysis
pandas==2.1.0
numpy==1.24.3

# Machine learning and recommendation algorithms
scikit-learn==1.3.0
surprise==1.1.3  # Optional (for advanced collaborative filtering)

# Flask API development
flask==3.1.0

# Visualization and exploratory data analysis (EDA)
matplotlib==3.7.2
seaborn==0.12.2

# Jupyter Notebook for EDA and experimentation
jupyter==1.0.0

# Additional utilities
python-dotenv==1.0.0  # Optional (for managing environment variables)
