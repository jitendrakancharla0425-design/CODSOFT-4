import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

cv = CountVectorizer()
matrix = cv.fit_transform(movies["Genre"])

similarity = cosine_similarity(matrix)

movie_name = input("Enter movie name: ")

if movie_name in movies["Movie"].values:

    index = movies[movies["Movie"] == movie_name].index[0]

    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")

    for i in scores[1:4]:
        print(movies.iloc[i[0]]["Movie"])

else:
    print("Movie not found!")
