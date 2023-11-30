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
    st.dataframe(df.head(20))
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
    movie_keywords = pd.read_csv('C:/Users/nicol/Downloads/keywords.csv')
    st.dataframe(movie_keywords.head())
    st.write("Columnas:")
    st.code(movie_keywords.columns)
    st.write("Shape:")
    st.code(movie_keywords.shape)

with st.expander("Credits"):
    movie_credits = pd.read_csv('C:/Users/nicol/Downloads/credits.csv')
    st.dataframe(movie_credits.head())
    st.write("Columnas:")
    st.code(movie_credits.columns)
    st.write("Shape:")
    st.code(movie_credits.shape)

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

# Function to extract genres from the 'genres' column
def extract_genres(genres_str):
    genres_list = ast.literal_eval(genres_str)
    return [genre['name'] for genre in genres_list if 'name' in genre]

# Apply the function to each row in the 'genres' column
df['genres'] = df['genres'].apply(extract_genres)

# Create the new DataFrame gen_df
gen_df = df.copy()
def build_chart(genre, cantidad, percentile=0.90):
    df = gen_df[gen_df['genres'].apply(lambda x: genre in x)]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(cantidad)
    
    return qualified

dataframe = None
list_genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Science Fiction', 'Fantasy', 'Animation','Foreign','Mystery','Family','Adventure', 'Documentary']
session.options = st.multiselect(label="Select Genre", options=list_genres)
selected = ', '.join(session.options)

st.text("")
st.text("")

session.slider_count = st.slider(label="movie_count", min_value=5, max_value=50)

st.text("")
st.text("")

buffer1, col1, buffer2 = st.columns([1.45, 1, 1])

is_clicked = col1.button(label="Recommend")

if is_clicked:
    dataframe = build_chart(str(selected), session.slider_count)

st.text("")
st.text("")
st.text("")

if dataframe is not None:
    st.table(dataframe) 

st.markdown(
    """
    ### Recomendación basada en modelos

    En lugar de depender únicamente de la similitud entre usuarios o elementos, como en los sistemas de recomendación basados en 
    usuario o en contenido, un sistema de recomendación basado en modelo crea un modelo matemático a partir de los datos históricos 
    de interacciones usuario-elemento. Este modelo se entrena para hacer predicciones sobre la preferencia del usuario para elementos 
    no vistos anteriormente.

    """
)

from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise import NormalPredictor

df = df.drop([19730, 29503, 35587]) #estos tienen fechas en los id's
df.reset_index(inplace=True, drop=True)

# seleccionamos los rating de las pelicuals que estan clasificadas segun el weighted avg
ratings_small = ratings_small[ratings_small['movieId'].isin(qualified_ids)]
ratings_small.reset_index(inplace=True, drop=True)

# Initialize a surprise reader object
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5), skip_lines=1)

# Load the data
data = Dataset.load_from_df(ratings_small[['userId','movieId','rating']], reader=reader)

random = NormalPredictor()

from surprise.model_selection import GridSearchCV

param_grid = {"n_epochs": [10, 15, 20], "lr_all": [0.002, 0.005, 0.001], "reg_all": [0.2, 0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)
gs.fit(data)

# Train a new model with the best parameters
best_params = gs.best_params['rmse']
best_svd = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
trainset = data.build_full_trainset()
best_svd.fit(trainset)
best_svd.predict(uid=3,iid=2959,r_ui=5.0)

with st.expander("Algoritmos"):
    st.markdown(
        """
        ## NormalPredictor

        NormalPredictor es un algoritmo simple de recomendación que realiza predicciones basadas en la distribución normal de las calificaciones. 
        Su enfoque es bastante básico y asigna calificaciones aleatorias según la media y la desviación estándar de las calificaciones existentes.

        ## Coclustering

        Coclustering es un algoritmo de filtrado colaborativo que agrupa tanto usuarios como elementos en clústeres.
        Funciona identificando patrones de comportamiento en función de cómo los usuarios califican los elementos. 
        A través de la agrupación, puede realizar recomendaciones para un usuario en función de la actividad de usuarios similares en el mismo clúster.

        ## SVD (Descomposición de Valor Singular)

        SVD es un algoritmo de filtrado colaborativo que utiliza la descomposición de valor singular para factorizar la matriz de interacciones 
        usuario-elemento en matrices de usuarios y elementos latentes. Esto permite capturar las relaciones latentes entre usuarios y elementos, 
        lo que facilita la predicción de calificaciones para elementos no vistos.

        En resumen, cada uno de estos algoritmos aborda el problema de recomendación de manera diferente, 
        ya sea asignando calificaciones de manera aleatoria (NormalPredictor), agrupando usuarios y elementos (Coclustering) o utilizando descomposición 
        de valor singular para identificar patrones latentes (SVD).
        """
    )
st.text('')
st.markdown(
        """
        'Metricas de los modelos ajustados'
        """
    )
st.image("./imagenes/modelos.jpg", use_column_width=True)

def get_recommendations(data, movie_metadata, user_id, top_n, algo):
    recommendations = []

    # Ensure 'id' in movie_metadata is integer (or the same type as item_id in your algorithm)
    movie_metadata['id'] = movie_metadata['id'].astype(int)

    # Create user-item interactions matrix
    user_movie_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')

    # Find movies the user hasn't interacted with
    non_interacted_movies = user_movie_interactions_matrix.loc[user_id][user_movie_interactions_matrix.loc[user_id].isnull()].index.tolist()

    for item_id in non_interacted_movies:
        # Predict the rating
        est = algo.predict(user_id, item_id).est

        # Find the movie title
        matched_movies = movie_metadata[movie_metadata['id'] == item_id]['title']

        if not matched_movies.empty:
            movie_name = matched_movies.iloc[0]
        else:
            movie_name = 'Unknown'

        recommendations.append((movie_name, est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

st.text('')
st.text('')
st.text('')

st.markdown(
    '''
    #### Recomendaciones para un usuario seleccionado (en base a sus ratings pasados)
    '''
)
user_id_input = st.number_input("Elegi un usuario entre [1 - 500]",
                            min_value=1,
                            max_value=500,
                            value=5,
                            step=1)
st.write(user_id_input)

# Obtener las recomendaciones para el usuario ingresado
user_predictions = get_recommendations(data = ratings_small, movie_metadata=df, user_id=user_id_input, top_n=10, algo=best_svd)

st.table(user_predictions)

st.write('')

st.markdown(
    '''
    ### Content-Based Filtering

    El modelo Content-Based es un enfoque de recomendación que se basa en las características intrínsecas de los elementos, como películas, 
    para realizar recomendaciones. En este caso, se utilizarán el título, la descripción y los tags de las películas como características 
    clave.

    '''
)
with st.expander("Cómo funciona el modelo Content-Based"):
    st.markdown(
        """
        #### Transformación de Datos

        Antes de aplicar la similitud de coseno, se realizan transformaciones en los datos para representar de manera efectiva las características. 
        Esto puede incluir la tokenización del texto, la eliminación de stop words y la creación de vectores que representen las características relevantes.

        #### Similitud de Coseno

        La similitud de coseno es una métrica que mide la similitud direccional entre dos vectores. En el contexto de la recomendación de películas, 
        se utiliza para medir la similitud entre los vectores que representan el contenido de diferentes películas. Cuanto más cercano a 1 es el valor de 
        similitud de coseno, más similar se considera el contenido.

        #### Características Utilizadas

        - *Título:* Representa la esencia de la película.
        - *Descripción:* Proporciona detalles adicionales sobre el contenido.
        - *Tags:* Etiquetas que describen aspectos específicos de la película.

        #### Proceso de Recomendación

        Cuando un usuario busca una película, el modelo Content-Based busca películas similares calculando la similitud de coseno entre el vector 
        que representa la película consultada y los vectores que representan otras películas en la base de datos. Las películas con mayor similitud 
        son recomendadas al usuario.

        En resumen, el modelo Content-Based utiliza el contenido intrínseco de las películas, como título, descripción y tags, y la similitud de coseno 
        para proporcionar recomendaciones basadas en la similitud de características.
        """
    )

import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

movie_md = movie_md[movie_md['vote_count']>=55]
movie_md = movie_md[['id','original_title','overview','genres']]
movie_md['title'] = movie_md['original_title'].copy()
movie_md.reset_index(inplace=True, drop=True)
movie_md.head()
movie_credits = movie_credits[['id','cast']]
movie_md = movie_md[movie_md['id'].str.isnumeric()]
movie_md['id'] = movie_md['id'].astype(int)
df = pd.merge(movie_md, movie_keywords, on='id', how='left')
df.reset_index(inplace=True, drop=True)
df = pd.merge(df, movie_credits, on='id', how='left')
df.reset_index(inplace=True, drop=True)
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in eval(x)])
df['genres'] = df['genres'].apply(lambda x: ' '.join([i.replace(" ","") for i in x]))
df['keywords'].fillna('[]', inplace=True)
df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])
df['keywords'] = df['keywords'].apply(lambda x: ' '.join([i.replace(" ",'') for i in x]))
df['cast'].fillna('[]', inplace=True)
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in eval(x)])
df['cast'] = df['cast'].apply(lambda x: ' '.join([i.replace(" ",'') for i in x]))
df['tags'] = df['overview'] + ' ' + df['genres'] +  ' ' + df['original_title'] + ' ' + df['keywords'] + ' ' + df['cast']
df.drop(columns=['genres','overview','original_title','keywords','cast'], inplace=True)
df.isnull().sum()
df.drop(df[df['tags'].isnull()].index, inplace=True)
df.drop_duplicates(inplace=True)

tfidf = TfidfVectorizer(max_features=5000)
vectorized_data = tfidf.fit_transform(df['tags'].values)
vectorized_dataframe = pd.DataFrame(vectorized_data.toarray(), index=df['tags'].index.tolist())

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=3000)
reduced_data = svd.fit_transform(vectorized_dataframe)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(reduced_data)

def recommendation(movie_title, cantidad):
    id_of_movie = df[df['title']==movie_title].index[0]
    distances = similarity[id_of_movie]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:cantidad+1]

    for i in movie_list:
        print(df.iloc[i[0]].title)

st.text("")
st.text('Selecciona una pelicula de tu interes')
peli_elegida=st.selectbox('Select Box',options=df['title'].unique())
peli_selected_str = ', '.join(session.options)

st.dataframe(recommendation(peli_selected_str, 20))

