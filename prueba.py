import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from wordcloud import WordCloud, STOPWORDS
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit import session_state as session

st.title("""
Movies Recommendation System
En el siguiente trabajo se buscara realizar un sistema de recomendación utilizando los elementos vistos en la materia.
 """)

# Subtitulos
st.subheader("TP2 Predictivo Avanzado")
st.subheader("Justo Juani Nico")

# Business Case
st.markdown(
    """
    ---
    ## Business Case
    
    Los modelos de recomendación de películas juegan un papel fundamental en la industria del entretenimiento.
    Estos modelos utilizan técnicas avanzadas de aprendizaje automático para analizar el historial de visualización
    de usuarios y proporcionar recomendaciones personalizadas. El objetivo es mejorar la experiencia del usuario,
    aumentar la retención y facilitar el descubrimiento de contenido relevante.
    
    Nuestro modelo se basa en datos predictivos avanzados para ofrecer recomendaciones precisas y adaptadas a
    las preferencias individuales de los usuarios. Exploraremos cómo estas recomendaciones pueden influir
    positivamente en la satisfacción del usuario y, por ende, en el éxito de la plataforma de streaming.
    """
)

st.markdown(
    """
    ---
    ## Bases de datos
    """
)
with st.expander("Movies_metadata"):
    df = pd.read_csv('C:/Users/nicol/Downloads/movies_metadata.csv')
    movie_md = pd.read_csv("C:/Users/nicol/Downloads/movies_metadata.csv")
    st.dataframe(df.head())
    st.write("Columnas:")
    st.code(df.columns)
    st.write("Shape:")
    st.code(df.shape)

with st.expander("Links"):
    links = pd.read_csv('C:/Users/nicol/Downloads/links.csv')
    st.dataframe(links.head())
    st.write("Columnas:")
    st.code(links.columns)
    st.write("Shape:")
    st.code(links.shape)

with st.expander("ratings_small"):
    ratings_small = pd.read_csv('C:/Users/nicol/Downloads/ratings_small.csv')
    st.dataframe(ratings_small.head())
    st.write("Columnas:")
    st.code(ratings_small.columns)
    st.write("Shape:")
    st.code(ratings_small.shape)
    
with st.expander("Keywords"):
    keywords = pd.read_csv('C:/Users/nicol/Downloads/keywords.csv')
    st.dataframe(keywords.head())
    st.write("Columnas:")
    st.code(keywords.columns)
    st.write("Shape:")
    st.code(keywords.shape)

with st.expander("Credits"):
    credits = pd.read_csv('C:/Users/nicol/Downloads/credits.csv')
    st.dataframe(credits.head())
    st.write("Columnas:")
    st.code(credits.columns)
    st.write("Shape:")
    st.code(credits.shape)

df = df.drop(['imdb_id'], axis=1)
df[df['original_title'] != df['title']][['title', 'original_title']].head()
df = df.drop('original_title', axis=1)
#porcentaje de peliculas sin revenue
#(df[df['revenue'] == 0].shape[0] / df.shape[0]) * 100
#lo reemplaon por nan asi no se lo confunde con que gano 0 plata
df['revenue'] = df['revenue'].replace(0, np.nan)
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
#(df[df['budget'].isnull()].shape[0] / df.shape[0]) * 100
df['return'] = df['revenue'] / df['budget']
#df[df['return'].isnull()].shape
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
#df['adult'].value_counts()
#como solo hay 9 la borramos
df = df.drop('adult', axis=1)
df['title'] = df['title'].astype('str')
df['overview'] = df['overview'].astype('str')
df['runtime'] = df['runtime'].astype('float')


st.markdown(
    """
    ---
    ## Feature Engineering
    """
)

st.code('''df = df.drop(['imdb_id'], axis=1)

df[df['original_title'] != df['title']][['title', 'original_title']].head()

df = df.drop('original_title', axis=1)

#porcentaje de peliculas sin revenue
#(df[df['revenue'] == 0].shape[0] / df.shape[0]) * 100

#lo reemplaon por nan asi no se lo confunde con que gano 0 plata
df['revenue'] = df['revenue'].replace(0, np.nan)

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
#(df[df['budget'].isnull()].shape[0] / df.shape[0]) * 100

df['return'] = df['revenue'] / df['budget']
#df[df['return'].isnull()].shape

df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#df['adult'].value_counts()

#como solo hay 9 la borramos
df = df.drop('adult', axis=1)

df['title'] = df['title'].astype('str')
df['overview'] = df['overview'].astype('str')
df['runtime'] = df['runtime'].astype('float')''')

st.markdown(
    """
    ---
    ## EDA
    """
)

st.markdown(
    """
    ### Title WordCloud
    """
)
st.image("./imagenes/words.jpg", use_column_width=True)

st.text("")

st.markdown(
    """
    <br>
    ### Paises y Lenguajes mas comunes
    """
)
st.image("./imagenes/paises.jpg", use_column_width=True)

st.text("")


st.markdown(
    """
    ### Popularidad, votos promedios y cantidad de votos
    """
)
st.image("./imagenes/votos.jpg", use_column_width=True)

st.markdown(
    """
    #### Peliculas mas populares
    """
)
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
st.dataframe(df[['title', 'popularity', 'year']].sort_values('popularity', ascending=False).head(10))

st.markdown(
    """
    #### Peliculas más votadas por los criticos
    """
)
st.dataframe(df[['title', 'vote_count', 'year']].sort_values('vote_count', ascending=False).head(10))

st.markdown(
    """
    ### Cantidad de peliculas por año
    """
)
st.image("./imagenes/ano.jpg", use_column_width=True)

st.markdown(
    """
    ### Distribucion de la duracion de las peliculas
    """
)
st.image("./imagenes/duracion.jpg", use_column_width=True)

st.markdown(
    """
    ### Peliculas mas taquilleras
    """
)
st.dataframe(df[(df['return'].notnull()) & (df['budget'] > 5e6)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return', ascending=False).head(10))

st.markdown(
    """
    ### Cantidad de peliculas por Género
    """
)
st.image("./imagenes/genero.jpg", use_column_width=True)

st.markdown(
    """
    ### Distribución de los retornos por Género    
    """
)
st.image("./imagenes/distGen.jpg", use_column_width=True)

st.markdown(
        """
        ## Sistemas de Recomendación

        - Recomendación Simple
        - Model Based Recomendation
        - User-Based 
        - Item-Based
        - Content-Based
        """
    )

st.markdown(
        """
        ### Recomendación Simple
        """
    )

from ast import literal_eval

vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.80)

df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['id','title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified_ids = [int(x) for x in qualified['id'].values]

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

# Top Movies
st.write("Las mejores 15 peliculas para recomendar (a nivel general) son:")
st.dataframe(qualified.head(15))
s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_df = df.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.90):
    df = gen_df[gen_df['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    return qualified

hola =build_chart('Drama').head(15)
st.dataframe(df.head(15))
st.dataframe(gen_df.head(15))

""" dataframe = None
list_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family']
session.options = st.multiselect(label="Select Genre", options=list_genres)

st.text("")
st.text("")

session.slider_count = st.slider(label="movie_count", min_value=5, max_value=50)

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

is_clicked = col1.button(label="Recommend")

if is_clicked:
    dataframe = build_chart(session.options, session.slider_count)

st.text("")
st.text("")
st.text("")

if dataframe is not None:
    st.table(dataframe) """