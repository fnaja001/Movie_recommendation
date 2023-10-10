

import pandas as pd
import numpy as np

# Read the movies data from a CSV file
movies = df = pd.read_csv('movies.csv')
movies.head()

# Display information about the DataFrame
df.info()

# Print the number of null values in each column
print(df.isnull().sum())

import re

def clean_title(title):
    # Remove special characters and symbols from the title
    return re.sub('[^a-zA-Z0-9 ]', "", title)

# Apply the clean_title function to the 'title' column and create a new column called 'clean_title'
movies['clean_title'] = movies['title'].apply(clean_title)
movies

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer object with ngram range (1,2)
vectorizer = TfidfVectorizer(ngram_range=(1,2))

# Apply TF-IDF vectorization to the 'clean_title' column
tfidf = vectorizer.fit_transform(movies['clean_title'])

from sklearn.metrics.pairwise import cosine_similarity

def search(title):
    # Clean the title using the clean_title function
    title = clean_title(title)
    
    # Transform the cleaned title into a vector using the vectorizer
    query_vec = vectorizer.transform([title])
    
    # Calculate the cosine similarity between the query vector and the TF-IDF matrix
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    
    # Get the indices of the top 5 most similar movies
    indices = np.argpartition(similarity, -5)[-5:]
    
    # Get the recommended movies based on the indices
    results = movies.iloc[indices][::-1]
    
    return results

import ipywidgets as widgets
from IPython.display import display

# Create a text input widget for the movie title
movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

# Create an output widget to display the search results
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

# Observe changes in the movie_input widget and call the on_type function
movie_input.observe(on_type, names='value')

# Display the movie_input widget and the movie_list widget
display(movie_input, movie_list)

# Read the ratings data from a CSV file
ratings = pd.read_csv('ratings.csv')
ratings

# Display the data types of each column in the ratings DataFrame
ratings.dtypes

movie_id = 1

# Get the unique users who rated the movie with the given movie_id and rating greater than or equal to 5
similar_users = ratings[(ratings['movieId'] == movie_id) & (ratings['rating'] >= 5)]['userId'].unique()
similar_users

# Get the movies recommended by similar users, who also rated the movies with a rating greater than 4
similar_user_recs = ratings[(ratings['userId'].isin(similar_users)) & (ratings['rating'] > 4)]['movieId']
similar_user_recs

# Calculate the percentage of similar users who recommended each movie
similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
similar_user_recs = similar_user_recs[similar_user_recs > 0.1]
similar_user_recs

# Get all users who rated the recommended movies with a rating greater than 4
all_users = ratings[(ratings['movieId'].isin(similar_user_recs.index)) & (ratings['rating'] > 4)]
all_users

# Calculate the percentage of all users who recommended each movie
all_users_recs = all_users['movieId'].value_counts() / len(all_users['userId'].unique())
all_users_recs

# Concatenate the two percentage series into a DataFrame
rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
rec_percentages.columns = ['similar', 'all']
rec_percentages

# Calculate the score based on the ratio of similar users to all users
rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']
rec_percentages = rec_percentages.sort_values('score', ascending=False)
rec_percentages

# Merge the top 10 movies with the highest scores with the movies DataFrame based on the movieId
rec_movies = rec_percentages.head(10).merge(movies, left_index=True, right_on='movieId')

import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def search(movie_title, movies):
    # Filter movies based on the input title
    filtered_movies = movies[movies['title'].str.contains(movie_title, case=False)]
    
    return filtered_movies

def find_similar_movies(movie_id, ratings, movies):
    # Filter ratings for movies with a high rating (greater than 4)
    high_rated_movies = ratings[(ratings['movieId'] == movie_id) & (ratings['rating'] > 4)]
    
    # Get unique users who rated the movie positively
    similar_users = high_rated_movies['userId'].unique()
    
    # Filter ratings by similar users who also rated other movies highly
    similar_user_ratings = ratings[(ratings['userId'].isin(similar_users)) & (ratings['rating'] > 4)]
    
    # Calculate the percentage of users who liked each movie
    similar_user_movie_percentages = similar_user_ratings['movieId'].value_counts() / len(similar_users)
    
    # Filter movies with a recommendation percentage greater than 10%
    recommended_movies = similar_user_movie_percentages[similar_user_movie_percentages > 0.10]
    
    # Get all users who rated the recommended movies highly
    all_users = ratings[(ratings['movieId'].isin(recommended_movies.index)) & (ratings['rating'] > 4)]
    
    # Calculate the percentage of users who liked each recommended movie among all users
    all_users_movie_percentages = all_users['movieId'].value_counts() / len(all_users['userId'].unique())
    
    # Combine the two percentage series into a DataFrame
    rec_percentages = pd.concat([recommended_movies, all_users_movie_percentages], axis=1)
    rec_percentages.columns = ['similar', 'all']
    
    # Calculate the score based on the ratio of similar users to all users
    rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']
    
    # Sort by score in descending order
    rec_percentages = rec_percentages.sort_values('score', ascending=False)
    
    # Merge with movie information and select relevant columns
    recommendations = rec_percentages.head(10).merge(movies, left_index=True, right_on='movieId')[['score', 'title', 'genres']]
    
    return recommendations

# Create a text input widget for the movie title
movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)

# Create an output widget to display the recommendations
recommendation_list = widgets.Output()

def on_type(change):
    with recommendation_list:
        recommendation_list.clear_output()
        title = change.new
        if len(title) > 5:
            results = search(title, movies)
            if not results.empty:
                movie_id = results.iloc[0]['movieId']
                display(find_similar_movies(movie_id, ratings, movies))

# Observe changes in the movie_name_input widget and call the on_type function
movie_name_input.observe(on_type, names='value')

# Display the movie_name_input widget and the recommendation_list widget
display(movie_name_input, recommendation_list)