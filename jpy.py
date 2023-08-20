# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# %%
movies = pd.read_csv('DataSets/tmdb_5000_movies.csv')
credits = pd.read_csv('DataSets/tmdb_5000_credits.csv')

# %%
movies.head(2)

# %%
movies.shape

# %%
credits.head()

# %%
movies = movies.merge(credits,on='title')

# %%
movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)

# %%
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# %%
movies.head()

# %%
import ast

# %%
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

# %%
movies.dropna(inplace=True)

# %%
movies['genres'] = movies['genres'].apply(convert)
movies.head()

# %%
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

# %%
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

# %%
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

# %%
movies['cast'] = movies['cast'].apply(convert)
movies.head()

# %%
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])

# %%
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

# %%
movies['crew'] = movies['crew'].apply(fetch_director)

# %%
#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)

# %%
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

# %%
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# %%
movies.head()

# %%
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# %%
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# %%
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()

# %%
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

# %%
new['tags'][0]  # all the tags listed in one string

# %%
import nltk

# %%
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# %%
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# %%
ps.stem('loved')     # converts words in base form [loving,loved,loves] to 'love'
# so that the tags like action and actions are not considered different. 

# %%
new['tags'] = new['tags'].apply(stem)

# %%
new['tags'][0]  # all the tags listed in one string

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')   # max tag words we consider are 5000.
# stop words in 'english' are [ 'is', 'to ','are'] etc. its not useful get simmilarity on those so we remove it.    

# %%
vector = cv.fit_transform(new['tags']).toarray()

# %%
vector.shape    # (x,y)  x vectors of y dimentions.

# %%
cv.get_feature_names_out()

# %%
from sklearn.metrics.pairwise import cosine_similarity

# %%
similarity = cosine_similarity(vector)

# %%
similarity

# %%
new[new['title'] == 'The Lego Movie'].index[0]

# %%
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        
    

# %%
recommend('Gandhi')

# %%
import pickle

# %%
# pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

# %%
# new.to_dict()

# %%
pickle.dump(new.to_dict(),open('movie_dict.pkl','wb'))

# %%



