import pandas as pd
import ast
import pickle
from pathlib import Path
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, model_dir="model_artifacts"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.movies = None
        self.tfidf = None
        self.similarity = None
        self.title_to_index = None

    # --------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------
    def fit_from_raw(self, movies_csv, credits_csv):
        movies = pd.read_csv(movies_csv)
        credits = pd.read_csv(credits_csv)

        # Try correct merge on IDs first
        if "id" in movies.columns and "movie_id" in credits.columns:
            df = movies.merge(credits, left_on="id", right_on="movie_id")
        else:
            df = movies.merge(credits, on="title")

        df = self._preprocess(df)
        self._build_model(df)
        self._save()

    def load(self):
        self.movies = self._load_pickle("movies.pkl")
        self.tfidf = self._load_pickle("tfidf.pkl")
        self.similarity = self._load_pickle("similarity.pkl")

        self.title_to_index = {t.lower(): i for i, t in enumerate(self.movies["title"])}

    def recommend(self, title, top_n=5):
        if self.movies is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        key = title.lower()

        if key not in self.title_to_index:
            key = self._fuzzy(key)
            if not key:
                return []

        idx = self.title_to_index[key]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

        return [(self.movies.iloc[i].title, float(score)) for i, score in scores]

    # --------------------------------------------------------------------
    # Internal Functions
    # --------------------------------------------------------------------
    def _preprocess(self, df):
        df = df[["title", "overview", "genres", "keywords", "cast", "crew"]]
        df.dropna(subset=["title", "overview"], inplace=True)

        df["soup"] = df.apply(self._create_soup, axis=1)
        return df.reset_index(drop=True)

    def _create_soup(self, row):
        def parse(obj, max_items=None, job=None):
            try:
                items = ast.literal_eval(obj)
            except:
                return []

            result = []
            for d in items:
                if job is None or d.get("job") == job:
                    result.append(d["name"].replace(" ", ""))
                    if max_items and len(result) >= max_items:
                        break
            return result

        overview = row["overview"]
        genres = " ".join(parse(row["genres"]))
        keywords = " ".join(parse(row["keywords"]))
        cast = " ".join(parse(row["cast"], max_items=3))
        director = " ".join(parse(row["crew"], job="Director"))

        return f"{overview} {genres} {keywords} {cast} {director}"

    def _build_model(self, df):
        self.movies = df
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=15000)
        tfidf_matrix = self.tfidf.fit_transform(df["soup"])
        self.similarity = cosine_similarity(tfidf_matrix)

        self.title_to_index = {t.lower(): i for i, t in enumerate(df["title"])}

    def _save(self):
        self._save_pickle(self.movies, "movies.pkl")
        self._save_pickle(self.tfidf, "tfidf.pkl")
        self._save_pickle(self.similarity, "similarity.pkl")

    def _save_pickle(self, obj, name):
        with open(self.model_dir / name, "wb") as f:
            pickle.dump(obj, f)

    def _load_pickle(self, name):
        with open(self.model_dir / name, "rb") as f:
            return pickle.load(f)

    def _fuzzy(self, key):
        matches = get_close_matches(key, self.title_to_index.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None
