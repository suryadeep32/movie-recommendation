#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv') 


# In[3]:


movies.head(2)


# In[4]:


movies.shape


# In[5]:


credits.head()

Merging two dataframes i.e "tmdb_5000_movies.csv" and 'tmdb_5000_credits.csv' on common column title
# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head()

Eliminating all the Unwanted Columns and only keeping the required ones i.e

movie_id, title, overview, genres, keywords, cast, crew.
# In[8]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


movies.head()


# In[13]:


movies.shape


# In[14]:


import ast

CONVERTING GENRES INTO READABLE FORMAT(used AST as dic keys are string so it has to be converted into list)
# In[15]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[16]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()

CONVERTING KEYWORDS INTO READABLE FORMAT
# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[18]:


def convert3(text):
    L = []
    counter=0
    for i in ast.literal_eval(text):
        if counter!=3:
            L.append(i['name']) 
            counter+=1
        else:
            break
    return L 


# In[19]:


movies['cast'] = movies['cast'].apply(convert3)
movies.head()

GETTING DIRECTORS OF MOVIE
# In[20]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L 


# In[21]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[22]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[23]:


movies.sample(5)


# In[26]:


movies['genres'].apply(lambda x:[i.replace(" ",'')for i in x])
movies['keywords'].apply(lambda x:[i.replace(" ",'')for i in x])
movies['cast'].apply(lambda x:[i.replace(" ",'')for i in x])
movies['crew'].apply(lambda x:[i.replace(" ",'')for i in x])


# In[27]:


movies.head()

CREATING TAGS ... WHICH COMPRISES ALL WORDS OF OVERVIEW, GENRE, KEYWORDS, CAST AND CREW
# In[28]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

CREATING NEW DATAFRAME FOR ONLY MOVIE, TITLE AND TAGS
# In[29]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()


# In[30]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
#list to string
new.head(8)


# In[31]:


new['tags']= new['tags'].apply(lambda x:x.lower())
#recommnended


# In[32]:


new.head()


# In[86]:


import nltk
# remove morphological affixes from words, leaving only the word stem


# In[34]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[35]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[36]:


new['tags'] = new['tags'].apply(stem)


# In[37]:


ps.stem('loving')

*VECTORIZATION*
# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[39]:


vector = cv.fit_transform(new['tags']).toarray()


# In[40]:


vector.shape


# In[72]:


#cv.get_feature_names()

CALCULATING SIMILARITY WITH COSINE OF VECTORS
# In[42]:


from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


similarity = cosine_similarity(vector)


# In[74]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[62]:


#new[new['title'] == 'Batman'].index[0]


# In[81]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    movie_lists=similarity[index]
    distances = sorted(list(enumerate(movie_lists)),reverse=True,key = lambda x: x[1])[1:6]
    for i in distances:
        print(new.iloc[i[0]].title)
        


# In[82]:


recommend('Tangled')


# In[85]:


recommend('Avatar')


# In[49]:





# In[87]:





# In[88]:





# In[ ]:





# In[89]:





# In[ ]:





# In[ ]:





# In[ ]:




