import numpy as np
import pandas as pd

# Load data
genome_scores = pd.read_csv('genome-scores.csv')
genome_tags = pd.read_csv('genome-tags.csv')
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

# Parse data
movies['name'] = movies.title.str[:-7]
movies['year'] = movies.title.str[-5:-1]
movies['genres_array'] = movies['genres'].str.split('|')

# Parse genres
for i, movie_genres in enumerate(movies["genres_array"]):
    if i / (len(movies) // 100 * 100) * 1000 % 10 == 0:
        print("{} %".format(i / (len(movies) // 100 * 100) * 100))
    for j, genres in enumerate(movie_genres):
        if genres in movies:
            movies.loc[i, genres] = 1
        if not genres in movies:
            movies[genres] = np.zeros((len(movies), 1), dtype = int)
            movies.loc[i, genres] = 1

# Calculate average rating from ratings
movie_rating = np.zeros(len(movies))
movie_rating_count = np.zeros(len(movies))
for i in range(len(movies)):
    if i / (len(movies) // 100 * 100) * 1000 % 10 == 0:
        print("{} %".format(i / (len(movies) // 100 * 100) * 100))
    movie_rating[i] = ratings[ratings.movieId == movies['movieId'][i]]['rating'].mean()
    movie_rating_count[i] = ratings[ratings.movieId == movies['movieId'][i]]['rating'].count()
    
movies['rating_mean'] = movie_rating
movies['rating_count'] = movie_rating_count

# Parse genome score
new_genome = pd.DataFrame(columns = genome_tags['tag'].T)
for i, movId in enumerate(movies['movieId']):
    if i / (len(movies) // 100 * 100) * 1000 % 10 == 0:
        print("{} %".format(i / (len(movies) // 100 * 100) * 100))
        
    if len(genome_scores['relevance'][genome_scores.movieId == movId].T) == 0:
        new_genome = new_genome.append(pd.DataFrame([np.zeros(len(genome_tags['tag'].T))], columns = genome_tags['tag'].T))
    elif len(genome_scores['relevance'][genome_scores.movieId == movId].T) == len(genome_tags['tag'].T):
        tmp = pd.DataFrame([genome_scores['relevance'][genome_scores.movieId == movId].T])
        tmp.columns = genome_tags['tag'].T
        new_genome = new_genome.append(tmp)

new_genome.set_index(movies["movieId"], inplace=True)
movies.set_index("movieId", inplace=True)
movies = movies.join(new_genome, how='inner')

movies.reset_index(inplace=True)

# Keep rating_count >= 50
movies = movies[movies.rating_count >= 50]
# Remove NAN
movies = movies.dropna()

movies.to_csv("PreprocessedData.csv")
