import streamlit as st
from src.recommender import MovieRecommender


@st.cache_resource(show_spinner=True)
def build_model():
    r = MovieRecommender(model_dir="model_artifacts")
    r.fit_from_raw("data/tmdb_5000_movies.csv", "data/tmdb_5000_credits.csv")
    r.load()
    return r


st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

st.write(
    "Content-based recommender using movie overview, genres, keywords, cast and "
    "director from the TMDB metadata."
)

model = build_model()

movie_title = st.text_input("Enter a movie title:", value="Avatar")
top_n = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

if st.button("Recommend"):
    with st.spinner("Searching..."):
        results = model.recommend(movie_title, top_n=top_n)

    if results:
        st.subheader("Recommendations")
        for i, (title, score) in enumerate(results, start=1):
            st.write(f"**{i}. {title}**  \nSimilarity score: `{score:.3f}`")
    else:
        st.error("No similar movies found. Try a different title or spelling.")
