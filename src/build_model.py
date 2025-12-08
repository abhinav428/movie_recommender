from recommender import MovieRecommender

if __name__ == "__main__":
    model = MovieRecommender()
    model.fit_from_raw("tmdb_5000_movies.csv", "tmdb_5000_credits.csv")
    print("Model built and artifacts saved!")
