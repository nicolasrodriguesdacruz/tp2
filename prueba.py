import ast
import pandas as pd
import numpy as np
import streamlit as st
from streamlit import session_state as session

st.title("""
Sistemas de Recomendacion para Peliculas
 """)

# Subtitulos
st.subheader("Segundo Trabajo Practico - Analisis Predictivo Avanzado")

# Business Case

st.markdown(
    """
    ---

### Importancia en la Industria del Entretenimiento
Los sistemas de recomendación de películas representan una herramienta esencial en la industria del entretenimiento. Estos sistemas, empleando algoritmos de aprendizaje automático, analizan patrones en el historial de visualización de los usuarios y ofrecen recomendaciones personalizadas. Al hacerlo, desempeñan un papel crucial en la mejora de la experiencia del usuario, aumentando la retención y facilitando el descubrimiento de nuevas películas y contenido relevante.

### Mejora de la Experiencia del Usuario
El enfoque de nuestro sistema está centrado en el usuario. Al proporcionar recomendaciones altamente relevantes, buscamos mejorar significativamente la experiencia de navegación y visualización en la plataforma. Esto incluye reducir el tiempo de búsqueda y aumentar la probabilidad de que el usuario encuentre películas que le gusten, mejorando así su satisfacción y lealtad hacia la plataforma.

### Impulso al Éxito de la Plataforma de Streaming
Las recomendaciones precisas y personalizadas tienen un impacto directo en el éxito de las plataformas de streaming. Al mejorar la experiencia del usuario, incrementamos la retención y fomentamos un mayor compromiso con la plataforma. Esto no solo se traduce en un aumento en el tiempo de visualización, sino también en una mayor probabilidad de que los usuarios recomienden la plataforma a otros, ampliando así la base de usuarios.

    """
)

st.markdown(
    """
    ---
    ## Bases de datos
    """
)

st.markdown(
    """
    ---
    ### Metadatos de Películas (`movies_metadata`)
    La base de datos `movies_metadata` incluye metadatos esenciales para 45,000 películas del conjunto completo de MovieLens, con películas lanzadas hasta julio de 2017. Esta base de datos es una pieza central en nuestro sistema, ya que contiene datos detallados como el reparto, equipo de producción, palabras clave de la trama, presupuesto, ingresos, pósters, fechas de lanzamiento, idiomas, países de producción y compañías productoras. Además, proporciona información valiosa sobre la votación y las calificaciones promedio en TMDB (The Movie Database).

    """
)

with st.expander("Movies_metadata"):
    df = pd.read_csv('Data/movies_metadata.csv')
    movie_md = pd.read_csv('Data/movies_metadata.csv')
    st.dataframe(df.head())
    st.write("Columnas:")
    st.code(df.columns)
    st.write("Shape:")
    st.code(df.shape)

st.markdown(
    """
    ---
    ### Enlaces (`links`) 
    La base de datos `links` contiene los identificadores de TMDB y IMDB para todas las películas presentes en el conjunto completo de MovieLens. Esta información es crucial para vincular nuestras películas con fuentes externas y enriquecer nuestras recomendaciones con datos adicionales disponibles en estas plataformas.

    """
)

with st.expander("Links"):
    links = pd.read_csv('Data/links.csv')
    st.dataframe(links.head())
    st.write("Columnas:")
    st.code(links.columns)
    st.write("Shape:")
    st.code(links.shape)

st.markdown(
    """
    ---
    ### Calificaciones Reducidas (`ratings_small`)
    `ratings_small` es un subconjunto del conjunto completo de MovieLens que incluye 100,000 calificaciones de 700 usuarios para 9,000 películas. Las calificaciones están en una escala de 1 a 5. Esta base de datos es esencial para entender las preferencias de los usuarios y para entrenar nuestros modelos de recomendación.

    """
)

with st.expander("ratings_small"):
    ratings_small = pd.read_csv('Data/ratings_small.csv')
    st.dataframe(ratings_small.head())
    st.write("Columnas:")
    st.code(ratings_small.columns)
    st.write("Shape:")
    st.code(ratings_small.shape)

st.markdown(
    """
    ---
    ### Palabras Clave (`keywords`) 
    La base de datos `keywords` contiene las palabras clave de la trama para las películas de MovieLens, presentadas en forma de objeto JSON. Las palabras clave ofrecen una visión más profunda de la trama y el contenido de las películas, lo que nos permite realizar recomendaciones más precisas y orientadas al contenido.

    """
)
    
with st.expander("Keywords"):
    keywords = pd.read_csv('Data/keywords.csv')
    st.dataframe(keywords.head())
    st.write("Columnas:")
    st.code(keywords.columns)
    st.write("Shape:")
    st.code(keywords.shape)

st.markdown(
    """
    ---
    ### Créditos (`credits`)
    `credits` incluye información sobre el reparto y el equipo de todas nuestras películas, también en formato JSON. Esta información nos permite considerar factores como actores y directores al hacer recomendaciones, lo que enriquece la personalización y relevancia de las sugerencias.

    """
)

with st.expander("Credits"):
    credits = pd.read_csv('Data/credits.csv')
    st.dataframe(credits.head())
    st.write("Columnas:")
    st.code(credits.columns)
    st.write("Shape:")
    st.code(credits.shape)

df = df.drop(['imdb_id'], axis=1)
df[df['original_title'] != df['title']][['title', 'original_title']].head()
df = df.drop('original_title', axis=1)
df['revenue'] = df['revenue'].replace(0, np.nan)
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
df['return'] = df['revenue'] / df['budget']
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
df = df.drop('adult', axis=1)
df['title'] = df['title'].astype('str')
df['overview'] = df['overview'].astype('str')
df['runtime'] = df['runtime'].astype('float')

def extract_genres(genres_str):
    genres_list = ast.literal_eval(genres_str)
    return [genre['name'] for genre in genres_list if 'name' in genre]

df['genres'] = df['genres'].apply(extract_genres)

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
st.write(' ')
st.markdown(
    """
    
    ### Title WordCloud
    """
)
st.image("imagenes/words.jpg", use_column_width=True)

st.write(' ')

st.markdown(
    """
    ### Paises y Lenguajes mas comunes
    """
)
st.image("imagenes/paises.jpg", use_column_width=True)

st.write(' ')

st.markdown(
    """
    ### Popularidad, votos promedios y cantidad de votos
    """
)
st.image("imagenes/votos.jpg", use_column_width=True)

st.write(' ')

def get_primary_genre(genres_list):
    if len(genres_list) == 0:
        return 'Unknown'
    else:
        return genres_list[0]

st.markdown(
    """
    #### Las 10 Peliculas mas populares
    """
)

st.write(' ')

df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
st.dataframe(df[['title', 'popularity', 'year', 'genres']].sort_values('popularity', ascending=False).head(10))

st.write(' ')

st.markdown(
    """
    #### Las 10 Peliculas más votadas por la audiencia
    """
)
st.dataframe(df[['title', 'vote_count', 'year', 'genres']].sort_values('vote_count', ascending=False).head(10))

st.write(' ')

st.markdown(
    """
    ### Cantidad de peliculas por año
    """
)
st.image("imagenes/ano.jpg", use_column_width=True)

st.write(' ')

st.markdown(
    """
    ### Distribucion de la duracion de las peliculas
    """
)
st.image("imagenes/duracion.jpg", use_column_width=True)

st.write(' ')

st.markdown(
    """
    ### Peliculas mas rentables
    """
)
st.dataframe(df[(df['return'].notnull()) & (df['budget'] > 5e6)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return', ascending=False).head(10))

st.write(' ')

st.markdown(
    """
    ### Cantidad de peliculas por Género
    """
)
st.image("imagenes/genero.jpg", use_column_width=True)

st.write(' ')

st.markdown(
    """
    ### Distribución de los retornos por Género    
    """
)
st.image("imagenes/distGen.jpg", use_column_width=True)

st.write(' ')

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
st.image("imagenes/wpg.jpg", use_column_width=True)

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
st.write("Las mejores 15 peliculas para recomendar (a nivel general)")
st.dataframe(qualified.head(15))


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

dataframee = None

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
    dataframee = build_chart(str(selected), session.slider_count)

st.text("")
st.text("")
st.text("")

if dataframee is not None:
    st.table(dataframee) 

## ACA VA EL IMPUT DEL USER

st.markdown(
    """
    ### Ingrese las peliculas que le gustan para que nuestro algoritmo le haga las recomendaciones    
    """
)

# Generate a unique user ID

input_user_id = 9999999

# Function to add a new user rating to the session state
def add_user_rating(title, rating, user_id):
    if 'input_ratings' not in st.session_state:
        st.session_state.input_ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    movie_id = movie_md[movie_md['title'] == title]['id'].iloc[0]
    
    # Create a new rating and append it if the title hasn't been rated yet
    if not ((st.session_state.input_ratings['userId'] == user_id) & 
            (st.session_state.input_ratings['movieId'] == movie_id)).any():
        new_rating = {'userId': user_id, 'movieId': movie_id, 'rating': rating}
        st.session_state.input_ratings = pd.concat(
            [st.session_state.input_ratings, pd.DataFrame([new_rating])], 
            ignore_index=True
        )

# Create a list of movie titles for the multi-select box
movie_titles = df['title'].dropna().unique()

# User interface for movie selection and rating
selected_titles = st.multiselect("Select movies:", movie_titles)

# Create a placeholder for the sliders
slider_container = st.container()

# Dictionary to hold the slider values
ratings = {}

# When movies are selected, create sliders for them
if selected_titles:
    for title in selected_titles:
        # Create a slider for each selected movie
        ratings[title] = slider_container.slider(f"Your rating for {title}:", min_value=1.0, max_value=5.0, step=0.1, value=2.5, key=title)

# Button to submit ratings
if st.button("Submit Ratings"):
    if selected_titles:
        for title in selected_titles:
            add_user_rating(title, ratings[title], input_user_id)
        st.success("You've successfully rated the selected movies.")
    else:
        st.error("Please select at least one movie to rate.")

# Display current ratings
if 'input_ratings' in st.session_state and not st.session_state.input_ratings.empty:
    st.write("Your current ratings:")
    st.dataframe(st.session_state.input_ratings)

# Concatenate the session state ratings with the original ratings dataframe
if 'input_ratings' in st.session_state and not st.session_state.input_ratings.empty:
    all_ratings = pd.concat([ratings_small, st.session_state.input_ratings], ignore_index=True)
else:
    all_ratings = ratings_small.copy()

# Convert movieId in all_ratings to integer
all_ratings['movieId'] = all_ratings['movieId'].astype(int)

# Assuming qualified_ids is a list, convert it to integers
qualified_ids = [int(id) for id in qualified_ids]

# Filter all_ratings to include only rows where movieId is in qualified_ids
all_ratings = all_ratings[all_ratings['movieId'].isin(qualified_ids)]


all_ratings.reset_index(inplace=True, drop=True)

if dataframe is not None:
    st.table(dataframe) 

st.markdown(
'''
### Model Based Recommender Systems
''')
st.write('En lugar de depender únicamente de la similitud entre usuarios o elementos, como en los sistemas de recomendación basados en usuario o en contenido, un sistema de recomendación basado en modelo crea un modelo matemático a partir de los datos históricos de interacciones usuario-elemento. Este modelo se entrena para hacer predicciones sobre la preferencia del usuario para elementos no vistos anteriormente.')

# Crear un expander para la explicación de los algoritmos
with st.expander("Algoritmos"):
    st.markdown(
        """
        ## NormalPredictor

        `NormalPredictor` es un algoritmo simple de recomendación que realiza predicciones basadas en la distribución normal de las calificaciones. 
        Su enfoque es bastante básico y asigna calificaciones aleatorias según la media y la desviación estándar de las calificaciones existentes.

        ## Coclustering

        `Coclustering` es un algoritmo de filtrado colaborativo que agrupa tanto usuarios como elementos en clústeres.
        Funciona identificando patrones de comportamiento en función de cómo los usuarios califican los elementos. 
        A través de la agrupación, puede realizar recomendaciones para un usuario en función de la actividad de usuarios similares en el mismo clúster.

        ## SVD (Descomposición de Valor Singular)

        `SVD` es un algoritmo de filtrado colaborativo que utiliza la descomposición de valor singular para factorizar la matriz de interacciones 
        usuario-elemento en matrices de usuarios y elementos latentes. Esto permite capturar las relaciones latentes entre usuarios y elementos, 
        lo que facilita la predicción de calificaciones para elementos no vistos.

        En resumen, cada uno de estos algoritmos aborda el problema de recomendación de manera diferente, 
        ya sea asignando calificaciones de manera aleatoria (`NormalPredictor`), agrupando usuarios y elementos (`Coclustering`) o utilizando descomposición 
        de valor singular para identificar patrones latentes (`SVD`).
        """
    )

st.image("imagenes/results_df.jpg", use_column_width=True)

from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
import joblib
from surprise import SVD, KNNBasic

df = df.drop([19730, 29503, 35587]) #estos tienen fechas en los id's
df.reset_index(inplace=True, drop=True)



# Initialize a surprise reader object
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5), skip_lines=1)

# Load the data
data = Dataset.load_from_df(all_ratings[['userId','movieId','rating']], reader=reader)

# seleccionamos los rating de las pelicuals que estan clasificadas segun el weighted avg

import json

# Load best parameters and initialize models with them
with open('Modelos/best_params_svd.json', 'r') as fp:
    best_params_svd = json.load(fp)

# Initialize the SVD model with the loaded parameters
best_svd = SVD(n_epochs=best_params_svd['n_epochs'], 
               lr_all=best_params_svd['lr_all'], 
               reg_all=best_params_svd['reg_all'])

# Load best parameters for user-based CF
with open('Modelos/best_params_user.json', 'r') as fp:
    best_params_user = json.load(fp)

# Initialize the user-based model with the loaded parameters
best_user_model = KNNBasic(k=best_params_user['k'], 
                           sim_options=best_params_user['sim_options'])

# Load best parameters for item-based CF
with open('Modelos/best_params_item.json', 'r') as fp:
    best_params_item = json.load(fp)

# Initialize the item-based model with the loaded parameters
best_item_model = KNNBasic(k=best_params_item['k'], sim_options=best_params_item['sim_options'])


# Assuming 'data' is your complete dataset including new ratings
trainset = data.build_full_trainset()

# Train SVD model
best_svd.fit(trainset)

# Train user-based CF model
best_user_model.fit(trainset)

# Train item-based CF model
best_item_model.fit(trainset)

import ast
import requests
import streamlit as st

# Función para extraer el género principal de la lista de géneros
def get_primary_genre(genres_list):
    if len(genres_list) == 0:
        return 'Unknown'
    else:
        return genres_list[0]

def fetch_movie_poster_url(movie_id):
    api_key = 'your_api_key'
    api_url = f"https://www.themoviedb.org/movie/{movie_id}"
    return api_url


def get_recommendations(data, movie_metadata, user_id, top_n, algo):
    recommendations = []
    movie_metadata['id'] = movie_metadata['id'].astype(int)

    user_movie_interactions_matrix = data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

    if user_id in user_movie_interactions_matrix.index:
        non_interacted_movies = user_movie_interactions_matrix.columns[user_movie_interactions_matrix.loc[user_id] == 0].tolist()

        for item_id in non_interacted_movies:

            est = algo.predict(user_id, item_id).est
            movie_details = movie_metadata[movie_metadata['id'] == item_id]

            if not movie_details.empty:
                movie = movie_details.iloc[0]
                movie_name = movie['title']
                vote_average = movie['vote_average']
                primary_genre = get_primary_genre(movie['genres'])
                duration = movie['runtime']
                poster_url = fetch_movie_poster_url(item_id)
                recommendations.append((movie_name, est, vote_average, primary_genre, duration, poster_url))
            else:
                recommendations.append(('Unknown', est, 'N/A', 'Unknown', 'N/A', 'default_poster_url'))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    else:
        return [("No data available for this user", 0, 'N/A', 'Unknown', 'N/A', 'default_poster_url')]



# Mostrar las recomendaciones y los pósters
user_predictions = get_recommendations(data=all_ratings, movie_metadata=df, user_id=input_user_id, top_n=10, algo=best_svd)
for movie_name, est, vote_average, primary_genre, duration, poster_url in user_predictions:
    st.write(f"Title: {movie_name}, Estimated Rating: {est}, Vote Average: {vote_average}, Primary Genre: {primary_genre}, Duration: {duration}")
    st.markdown(f"[![Mas Info]({poster_url})]({poster_url})")


st.markdown(
    """
    ### Memory based recommender system 


    Los métodos basados en memoria emplean datos históricos de calificaciones de usuarios para calcular la similitud entre usuarios o elementos (como películas o productos). La premisa de estos 
    métodos consiste en definir una medida de similitud entre usuarios o elementos y, a partir de esta, identificar aquellos más similares para recomendar elementos no vistos o no interactuados 
    previamente. Existen dos tipos principales de sistemas de recomendación basados en memoria:

    """
)

st.markdown(
 """
### Modelo User-Based en Recomendación de Películas
 """
)
st.write('El modelo User-Based, o basado en usuario, es un enfoque de recomendación que se centra en las preferencias de usuarios similares para realizar recomendaciones a un usuario específico.')
st.image("imagenes/USER-BASED-FILTERING.jpg", use_column_width=True)
with st.expander("Cómo funciona un Modelo User-Based en Recomendación de Películas"):
    st.markdown(
        """
        ### 1. Creación de la Matriz Usuario-Película

        Se construye una matriz que representa las calificaciones de los usuarios para diferentes películas. Cada fila de la matriz corresponde 
        a un usuario, cada columna a una película, y los valores en la matriz son las calificaciones dadas por los usuarios a las películas.

        ### 2. Medición de Similitud entre Usuarios

        Se calcula la similitud entre usuarios para identificar aquellos con preferencias similares. Comúnmente, se utiliza la similitud de 
        coseno o la correlación de Pearson. Cuanto más cercano a 1 es el valor de similitud, más similar se consideran dos usuarios.

        ### 3. Identificación de Vecinos más Cercanos

        Para un usuario específico, se identifican los usuarios más cercanos basándose en la similitud calculada. Estos usuarios se conocen como 
        "vecinos más cercanos" y comparten preferencias similares.

        ### 4. Generación de Recomendaciones

        Las recomendaciones se generan considerando las calificaciones de los vecinos más cercanos para películas que el usuario en cuestión aún no ha visto. 
        Esto se puede hacer promediando las calificaciones de los vecinos o utilizando métodos más avanzados.
        """
    )

best_user_model_predictions = get_recommendations(data=all_ratings, movie_metadata=df, user_id=input_user_id, top_n=10, algo=best_user_model)
for movie_name, est, vote_average, primary_genre, duration, poster_url in best_user_model_predictions:
    st.write(f"Title: {movie_name}, Estimated Rating: {est}, Vote Average: {vote_average}, Primary Genre: {primary_genre}, Duration: {duration}")
    st.markdown(f"[![Poster]({poster_url})]({poster_url})")


st.markdown(
 """
### Modelo Item-Based en Recomendación de Películas

 """
)
st.write('El modelo Item-Based, o basado en elementos, es un enfoque de recomendación que se centra en las características intrínsecas de los elementos, como las películas, para realizar recomendaciones.')
st.image("imagenes/ITEM-BASED-FILTERING.jpg", use_column_width=True)
with st.expander("Cómo funciona un Modelo Item-Based en Recomendación de Películas"):
    st.markdown(
        """
        ### 1. Creación de la Matriz Usuario-Película

        Similar al modelo User-Based, se construye una matriz que representa las calificaciones de los usuarios para diferentes películas. 
        Cada fila de la matriz corresponde a un usuario, cada columna a una película, y los valores en la matriz son las calificaciones dadas por los usuarios.

        ### 2. Transposición de la Matriz Usuario-Película

        En el modelo Item-Based, la matriz Usuario-Película se transpone, convirtiéndola en una matriz Película-Usuario. 
        Cada fila ahora representa una película, y cada columna representa un usuario.

        ### 3. Medición de Similitud entre Películas

        Se calcula la similitud entre películas para identificar aquellas con características similares. 
        Puede utilizarse la similitud de coseno, correlación de Pearson u otras métricas. 
        Cuanto más cercano a 1 es el valor de similitud, más similar se consideran dos películas.

        ### 4. Generación de Recomendaciones

        Cuando un usuario busca una película, el modelo Item-Based busca películas similares basándose en la similitud de características. 
        Las películas con mayor similitud se recomiendan al usuario.
        """
    )



best_item_model_predictions = get_recommendations(data=all_ratings, movie_metadata=df, user_id=input_user_id, top_n=10, algo=best_item_model)
for movie_name, est, vote_average, primary_genre, duration, poster_url in best_item_model_predictions:
    st.write(f"Title: {movie_name}, Estimated Rating: {est}, Vote Average: {vote_average}, Primary Genre: {primary_genre}, Duration: {duration}")
    st.markdown(f"[![Poster]({poster_url})]({poster_url})")

st.write('')

st.markdown(
    '''
    ### Content-Based Filtering

    El modelo Content-Based es un enfoque de recomendación que se basa en las características intrínsecas de los elementos, como películas, 
    para realizar recomendaciones. En este caso, se utilizarán el título, la descripción y los tags de las películas como características 
    clave.

    '''
)

st.image("imagenes/CONTENT-BASED-RECOM.jpg", use_column_width=True)

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

        """
    )


df = pd.read_csv('Data/df_content_based_mod.csv')

similarity = joblib.load("Modelos/similarity.pkl")

def recommendation(movie_title):
    # Find the index of the movie that matches the title
    id_of_movie = df[df['title'] == movie_title].index[0]
    
    # Get the similarity scores for all movies with the given movie
    distances = similarity[id_of_movie]
    
    # Sort the movies based on the similarity scores
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:15]
    
    # Extract the movie indices
    movie_indices = [i[0] for i in movie_list]
    
    # Create a DataFrame with movie details
    recommended_movies = df.iloc[movie_indices]
    
    # Add the similarity score to the DataFrame
    recommended_movies['similarity_score'] = [i[1] for i in movie_list]
    
    recommended_movies = recommended_movies.drop(['tags'], axis = 1)

    return recommended_movies


st.text("")
st.text('Selecciona una pelicula de tu interes')
peli_elegida=st.selectbox('Select Box',options=df['title'].unique())
peli_elegida_str = str(peli_elegida)

recommended_movies = (recommendation(peli_elegida_str))

st.table(recommended_movies)

st.markdown(
    """
    ---
    ### Conclusión
    En resumen, nuestro sistema de recomendación de películas representa un avance significativo en la forma en que las plataformas de entretenimiento interactúan con sus usuarios. A través de la personalización y la innovación tecnológica, buscamos no solo entretener, sino también crear una experiencia de usuario más rica y satisfactoria, contribuyendo al crecimiento y éxito continuo de la plataforma de streaming.
    """

)