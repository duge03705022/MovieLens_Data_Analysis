import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from apyori import apriori

# Load data
genome_tags = pd.read_csv('genome-tags.csv')
movies = pd.read_csv('PreprocessedData.csv', low_memory = False)
movies = movies.drop(labels=["Unnamed: 0"], axis="columns")

# Top 5 & Bottom 5 Movies
movie_top5 = movies.sort_values(by = "rating_mean", ascending = False).head(5)
print(movie_top5[["title", "rating_mean", "genres"]])
movie_bottom5 = movies.sort_values(by = "rating_mean", ascending = True).head(5)
print(movie_bottom5[["title", "rating_mean", "genres"]])

# Genres analysis
movies_genres = movies.drop(np.append(genome_tags["tag"], ["movieId", "title", "genres", "name", "year", "rating_mean", "rating_count", "genres_array"]), axis = 1)

movies_rating4 = movies[movies.rating_mean >= 4.0]
movies_rating4_genres = movies_rating4.drop(np.append(genome_tags["tag"], ["movieId", "title", "genres", "name", "year", "rating_mean", "rating_count", "genres_array"]), axis = 1)

movies_rating35 = movies[movies.rating_mean >= 3.5]
movies_rating35_genres = movies_rating35.drop(np.append(genome_tags["tag"], ["movieId", "title", "genres", "name", "year", "rating_mean", "rating_count", "genres_array"]), axis = 1)

print("========================================")
print("Rating over 4.0")
tmp = movies_rating4_genres.sum() / movies_genres.sum()
print(tmp.sort_values(ascending = False))

print("========================================")
print("Rating over 3.5")
tmp = movies_rating35_genres.sum() / movies_genres.sum()
print(tmp.sort_values(ascending = False))

# Scatter plot of rating_count & rating_mean
x = movies["rating_count"]
y = movies["rating_mean"]

plt.scatter(x, y, s = 1)
plt.xlabel('rating_count')
plt.ylabel('rating_mean')

plt.show()

# Correlation coefficient between genome_tags
corr = pd.DataFrame(movies[genome_tags["tag"]])
corr["rating_mean"] = movies["rating_mean"]
corrTable = corr.corr()

print("========================================")
print("Correlation coefficient - descending")
print(corrTable["rating_mean"].sort_values(ascending = False))

print("========================================")
print("Correlation coefficient - ascending")
print(corrTable["rating_mean"].sort_values(ascending = True))

# Relevance analysis
genres_array = list(movies["genres_array"])
association_rules = apriori(genres_array, min_support=0.01, min_confidence=0.33, min_lift=3) 
association_results = list(association_rules)

for item in association_results:
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
