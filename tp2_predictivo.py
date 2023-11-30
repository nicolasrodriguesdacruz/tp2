#from IPython.display import Image, HTML
#import json
#import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from scipy import stats

from wordcloud import WordCloud, STOPWORDS
#import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
#import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.corpus import wordnet


df = pd.read_csv('C:/Users/nicol/Downloads/movies_metadata.csv')
df.head()

data = df



#Feature ing

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

#EDA

title_corpus = ' '.join(df['title'])
overview_corpus = ' '.join(df['overview'])

title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()

#Paises y Lenguajes mas comunes

df['original_language'].drop_duplicates().shape[0]
lang_df = pd.DataFrame(df['original_language'].value_counts())
lang_df['language'] = lang_df.index
lang_df.columns = ['number', 'language']
plt.figure(figsize=(12,5))
sns.barplot(x='language', y='number', data=lang_df.iloc[0:11]) #dejamos a el ingles afuera pq tiene como 32k
plt.show()

lang_df['number'].sort_values(ascending = False)

#Popoularidad, voteos propedios y cuento de votos

def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan

df['popularity'] = df['popularity'].apply(clean_numeric).astype('float')
df['vote_count'] = df['vote_count'].apply(clean_numeric).astype('float')
df['vote_average'] = df['vote_average'].apply(clean_numeric).astype('float')
df['vote_average'] = df['vote_average'].replace(0, np.nan)

#df['popularity'].describe()

ax=df['popularity'].plot(logy=True, kind='hist', edgecolor='black', linewidth=1.2, alpha=0.7, color='skyblue', bins=20)
ax.set_xlabel('Popularidad')
ax.set_ylabel('Frequencia (log scale)')
ax.set_title('Frecuencia de popularidad de peliculas')

#Most Popular Movies by Popularity Score¶
df[['title', 'popularity', 'year']].sort_values('popularity', ascending=False).head(10)

#meter alguna infographic

#Most Voted on Movies por los criticos
df[['title', 'vote_count', 'year']].sort_values('vote_count', ascending=False).head(10)

#vote avg
df[df['vote_count'] > 2000][['title', 'vote_average', 'vote_count' ,'year']].sort_values('vote_average', ascending=False).head(10)

#meter alguna infographic

#podriamos hacer correlaciones aca

#Pelicials por ano

year_count = df.groupby('year')['title'].count()
plt.figure(figsize=(18,5))
evolucion_data=year_count.plot()

evolucion_data.set_xlabel('Año')
evolucion_data.set_ylabel('Cantidad')
evolucion_data.set_title('Cantidad de Datos por Año')

#Duracion pelis

#df[(df['runtime'] <= 0)]

#imputamos con la mediana los <= 0
mediana = df['runtime'].median()

df.loc[df['runtime'] <= 0, 'runtime'] = mediana

plt.figure(figsize=(12,6))

sns.distplot(df[(df['runtime'] < 1000)]['runtime'])

df['year'] = pd.to_numeric(df['year'], errors='coerce')

# df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year


# plt.figure(figsize=(18,5))
# year_runtime = df[df['year'] != 'NaT'].groupby('year')['runtime'].mean()
# plt.plot(year_runtime.index, year_runtime)

# # Plot the data
# plt.plot(year_runtime.index, year_runtime)


# # Extract the year part from the datetime objects
# start_year = year_runtime.index.min().year
# end_year = year_runtime.index.max().year
# # Generate ticks for every ten years
# ticks = pd.date_range(start=f'{start_year}', end=f'{end_year}', freq='10YS')

# # Set x-axis ticks
# plt.xticks(ticks)

# plt.show()

#pelicuals mas taquilleras"""

#df[(df['return'].notnull()) & (df['budget'] > 5e6)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return', ascending=False).head(10)

#meter infograpghic

 #Genero

#este código transforma una columna del DataFrame que contiene listas de géneros (en formato de diccionarios) en un formato donde cada género tiene su propia fila, y luego cuenta el número total de géneros únicos.


import ast

def convert_genre(entry):
    # If the entry is a string, try to convert it using ast.literal_eval
    if isinstance(entry, str):
        try:
            return [d['name'] for d in ast.literal_eval(entry)]
        except ValueError:
            # Handle cases where ast.literal_eval fails to parse the string
            return []
    # If the entry is already a list, extract the genre names
    elif isinstance(entry, list):
        return [d['name'] for d in entry if isinstance(d, dict) and 'name' in d]
    # If the entry is neither a string nor a list, return an empty list
    else:
        return []

# Apply the function to the 'genres' column
df['genres'] = df['genres'].fillna('[]').apply(convert_genre)

# Expanding the list of genre names into separate rows
s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'

# Joining the series of genre names with the original DataFrame without the 'genres' column
gen_df = df.drop('genres', axis=1).join(s)

#df['genres']

pop_gen = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()
pop_gen.columns = ['genre', 'movies']
pop_gen.head(15)

plt.figure(figsize=(18, 8))
sns.barplot(x='genre', y='movies', data=pop_gen.head(15))
plt.title('Top 15 Movie Genres by Popularity')  # Agrega tu título aquí
plt.show()

violin_genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Science Fiction', 'Fantasy', 'Animation']
violin_movies = gen_df[(gen_df['genre'].isin(violin_genres))]
plt.figure(figsize=(18,8))
#fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
#sns.boxplot(x='genre', y='return', data=violin_movies, palette="muted", ax =ax)
#ax.set_ylim([0, 10])
#plt.title('Distribución del Retorno por Género')
#plt.show()

# Sistemas de recomnadcion

### Recomendacion simple basada en IMBD's weighted average

from ast import literal_eval

vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.80)

df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['id','title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
#qualified.shape

qualified_ids = [int(x) for x in qualified['id'].values]
#qualified_ids

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

# Top Movies"""

qualified.head(15)

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

build_chart('Drama').head(15)


from surprise import Dataset, Reader

from surprise.prediction_algorithms.matrix_factorization import SVD

from surprise import accuracy

from surprise.model_selection import cross_validate

from surprise import NormalPredictor

df = df.drop([19730, 29503, 35587]) #estos tienen fechas en los id's
df.reset_index(inplace=True, drop=True)

links = pd.read_csv('C:/Users/nicol/Downloads/links.csv')

links.head()

ratings_small = pd.read_csv('C:/Users/nicol/Downloads/ratings_small.csv')

ratings_small.head()

#ratings_small.info()

# seleccionamos los rating de las pelicuals que estan clasificadas segun el weighted avg

ratings_small = ratings_small[ratings_small['movieId'].isin(qualified_ids)]
ratings_small.reset_index(inplace=True, drop=True)
#ratings_small.shape

# Initialize a surprise reader object
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5), skip_lines=1)

# Load the data
data = Dataset.load_from_df(ratings_small[['userId','movieId','rating']], reader=reader)

# df para los result
results_list = []
#print("Evaluating User-Based Baseline...")

random = NormalPredictor()

results = cross_validate(random, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

rmse = results['test_rmse'].mean()
mae = results['test_mae'].mean()

#print(f'Average RMSE: {rmse}')
#print(f'Average MAE: {mae}')
results_list += [{'Model': 'NormalPredictor', 'Average MAE': mae, 'Average RMSE': rmse}]
results_df = pd.DataFrame(results_list)
#print(results_df)

from surprise.model_selection import GridSearchCV

param_grid = {"n_epochs": [10, 15, 20], "lr_all": [0.002, 0.005, 0.001], "reg_all": [0.2, 0.4, 0.6]}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)

gs.fit(data)

# RMSE and MAE
rmse = gs.best_score['rmse'].mean()
mae = gs.best_score['mae'].mean()

#print(f'Average RMSE: {rmse}')
#print(f'Average MAE: {mae}')

#print(f'best parms: {gs.best_params["rmse"]}')

results_list += [{'Tipo sist recom': 'Model Based','Algoritmo': 'SVD','Average MAE': mae, 'Average RMSE': rmse }]
results_df = pd.DataFrame(results_list)

#results_df

# Train a new model with the best parameters
best_params = gs.best_params['rmse']
best_svd = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
trainset = data.build_full_trainset()
best_svd.fit(trainset)


best_svd.predict(uid=3,iid=2959,r_ui=5.0)

from surprise import CoClustering

# Co-Clustering
co_clustering = CoClustering()
results_co_clustering = cross_validate(co_clustering, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# RMSE and MAE
rmse = results_co_clustering['test_rmse'].mean()
mae = results_co_clustering['test_mae'].mean()

#print(f'Average RMSE: {rmse}')
#print(f'Average MAE: {mae}')


results_list += [{'Tipo sist recom': 'Model Based','Algoritmo': 'CoClustering','Average MAE': mae, 'Average RMSE': rmse }]
                               
results_df = pd.DataFrame(results_list)

#results_df


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

get_recommendations(data = ratings_small, movie_metadata=df, user_id=654, top_n=10, algo=best_svd)

# Memory Based Recommender System

#Memory-based methods use user rating historical data to compute the similarity between users or items. The idea behind these methods is to #define a similarity measure between users or items, and find the most similar to recommend unseen items. Memory based recommender systems are #of 2 types -

#Item-Bassed

# Content Based

#importamos librerias


import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# Read the movies meta-data(we will be using the feature genre, overview & title from this)
movie_md = pd.read_csv("C:/Users/nicol/Downloads/movies_metadata.csv")

# Read the keywords
movie_keywords = pd.read_csv("C:/Users/nicol/Downloads/keywords.csv")

# Read the credits
movie_credits = pd.read_csv("C:/Users/nicol/Downloads/credits.csv")

movie_md = movie_md[movie_md['vote_count']>=55]
movie_md = movie_md[['id','original_title','overview','genres']]
movie_md['title'] = movie_md['original_title'].copy()
movie_md.reset_index(inplace=True, drop=True)
movie_md.head()

#From movies metadata column we are going to work with the following features -
#Genres

#Original Title

#Overview

#id

#From movies keywords column we are going to work with the following features -
#keywords (to fetch the keywords)

#id (to merge dataframe)

#From movies credits column we are going to work with the following features -
#cast - To get the name of the actors

#id - To merge dataframe


movie_credits = movie_credits[['id','cast']]

###**Data Cleaning & Preprocessing**


# Removing the records for which the id is not available
movie_md = movie_md[movie_md['id'].str.isnumeric()]


# Merge all dataframe as a single entity
# To merge the ids must be of same datatype
movie_md['id'] = movie_md['id'].astype(int)

# Merge
df = pd.merge(movie_md, movie_keywords, on='id', how='left')

# Reset the index
df.reset_index(inplace=True, drop=True)

# Merge with movie credits
df = pd.merge(df, movie_credits, on='id', how='left')

# Reset the index
df.reset_index(inplace=True, drop=True)

#Let's fetch the genres, keywords, cast to vectorize them later

# Lets first start with cleaning the movies metadata
# Fetchin the genre list from the column
df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in eval(x)])

# Replaces spaces in between genre(ex - sci fi to scifi) and make it a string
df['genres'] = df['genres'].apply(lambda x: ' '.join([i.replace(" ","") for i in x]))

# Filling the numm values as []
df['keywords'].fillna('[]', inplace=True)

# Let's clean the keywords dataframe to extract the keywords
# Fetchin the keyword list from the column
df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])

# Remove the expty spaces and join all the keyword wwwith spaces
df['keywords'] = df['keywords'].apply(lambda x: ' '.join([i.replace(" ",'') for i in x]))

# Filling the numm values as []
df['cast'].fillna('[]', inplace=True)

# Let's clean the cast dataframe to extract the name of aactors from cast column
# Fetchin the cast list from the column
df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in eval(x)])

# Remove the expty spaces and join all the cast with spaces
df['cast'] = df['cast'].apply(lambda x: ' '.join([i.replace(" ",'') for i in x]))

###**Let's merge all content/description of movies as a single feature**


df['tags'] = df['overview'] + ' ' + df['genres'] +  ' ' + df['original_title'] + ' ' + df['keywords'] + ' ' + df['cast']

# Delete useless columns
df.drop(columns=['genres','overview','original_title','keywords','cast'], inplace=True)

df.isnull().sum()

df.drop(df[df['tags'].isnull()].index, inplace=True)

df.drop_duplicates(inplace=True)
#df.shape

### **Convert the contents to vectors**

#As our model will not be able to understand text inputs we would have to vectorize them and make it in form of machine readable format

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a tfidf object
tfidf = TfidfVectorizer(max_features=5000)

# Transform the data
vectorized_data = tfidf.fit_transform(df['tags'].values)

#vectorized_data

vectorized_dataframe = pd.DataFrame(vectorized_data.toarray(), index=df['tags'].index.tolist())

#Perform Dimension Reduction**

#We are gonna perform dimensional reduction as computing similarities with such huge dimensions would be exremely computationally expensive

from sklearn.decomposition import TruncatedSVD

# Initialize a PCA object
svd = TruncatedSVD(n_components=3000)

# Fit transform the data
reduced_data = svd.fit_transform(vectorized_dataframe)

# Print the shape
#reduced_data.shape

#svd.explained_variance_ratio_.cumsum()

###Recommendation

#Compute a similarity metric on vectors for recommendation**

#Now in order to make recommendations we would have to compute any similarity index ex- cosine similarity, eucledian distance, Jaccard distance, etc. here we are going to use cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(reduced_data)

#THE MAGIC IS COMING

def recommendation(movie_title, cantidad):
    id_of_movie = df[df['title']==movie_title].index[0]
    distances = similarity[id_of_movie]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:cantidad+1]

    for i in movie_list:
        print(df.iloc[i[0]].title)

recommendation('The Matrix', 10)

recommendation('Titanic', 15)

recommendation('Rocky', 20)

recommendation('Inception', 5)


from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.knns import KNNBasic

#User based
# Define the parameter grid
param_grid_user = {
    'k': [20, 30, 40],
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'],
        'user_based': [True]
    }
}

# Perform grid search
gs_user = GridSearchCV(KNNBasic, param_grid_user, measures=['RMSE', 'MAE'], cv=5, joblib_verbose=0)
gs_user.fit(data)

# Best score and parameters
#print(f'Best RMSE: {gs_user.best_score["rmse"]}')
#print(f'Best MAE: {gs_user.best_score["mae"]}')
#print(f'Best parameters: {gs_user.best_params["rmse"]}')

# Update results_df
results_df = pd.DataFrame()
results_list += [{'Tipo sist recom': 'Memory-based CF (User-Based)', 'Algoritmo': 'KNNBasic', 'Average MAE': gs_user.best_score["mae"], 'Average RMSE': gs_user.best_score["rmse"], 'Parameters': gs_user.best_params["rmse"]}]

results_df = pd.DataFrame(results_list)

best_params = gs_user.best_params['rmse']
best_user_model = KNNBasic(k=best_params['k'], sim_options=best_params['sim_options'])
best_user_model.fit(trainset)

# Display the updated DataFrame
#print(results_df)
get_recommendations(ratings_small, df, 671,10,best_user_model)

# Define the parameter grid
param_grid_item = {
    'k': [20, 30, 40],
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'],
        'user_based': [False]
    }
}

# Perform grid search
gs_item = GridSearchCV(KNNBasic, param_grid_item, measures=['RMSE', 'MAE'], cv=5, joblib_verbose=0)
gs_item.fit(data)

# Best score and parameters
#print(f'Best RMSE: {gs_item.best_score["rmse"]}')
#print(f'Best MAE: {gs_item.best_score["mae"]}')
#print(f'Best parameters: {gs_item.best_params["rmse"]}')

# Update results_df
results_list += [{'Tipo sist recom': 'Memory-based CF (Item-Based)', 'Algoritmo': 'KNNBasic', 'Average MAE': gs_item.best_score["mae"], 'Average RMSE': gs_item.best_score["rmse"], 'Parameters': gs_item.best_params["rmse"]}]

results_df = pd.DataFrame(results_list)

best_params = gs_item.best_params['rmse']
best_item_model = KNNBasic(k=best_params['k'], sim_options=best_params['sim_options'])
best_item_model.fit(trainset)

