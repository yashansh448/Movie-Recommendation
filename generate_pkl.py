import pandas as pd
import pickle
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading CSVs...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print("Merging datasets...")
movies = movies.merge(credits, left_on='id', right_on='movie_id')

# Keep only needed columns
movies = movies[['movie_id', 'title_x', 'overview', 'genres', 'keywords']].copy()
movies.rename(columns={'title_x': 'title'}, inplace=True)

# Parse genres and keywords from JSON strings
def parse_names(obj):
    try:
        return ' '.join([i['name'].replace(' ', '') for i in ast.literal_eval(obj)])
    except:
        return ''

movies['genres'] = movies['genres'].apply(parse_names)
movies['keywords'] = movies['keywords'].apply(parse_names)

# Fill missing overviews
movies['overview'] = movies['overview'].fillna('')

# Combine features into one text column
movies['tags'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords']

print("Computing TF-IDF and cosine similarity (this may take a minute)...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Saving movie_data.pkl...")
with open('movie_data.pkl', 'wb') as f:
    pickle.dump((movies, cosine_sim), f)

print("Done! movie_data.pkl has been regenerated successfully.")