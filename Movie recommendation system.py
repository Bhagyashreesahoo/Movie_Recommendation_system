#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv.zip")
credits=pd.read_csv("tmdb_5000_credits.csv.zip")


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.shape


# movies.head()

# In[7]:


movies.head()


# In[8]:


movies=movies[["movie_id","title","overview","genres","keywords","cast","crew"]]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


def convert(obj):
    lst=[]
    for i in ast.literal_eval(obj):
       lst.append(i['name'])
    return lst


# In[15]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[16]:


movies['genres']=movies['genres'].apply(convert)


# In[17]:


movies.head()


# In[18]:


movies['keywords']=movies['keywords'].apply(convert)


# In[19]:


def convert3(obj):
    l=[]
    counter =0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
            counter +=1
        else:
            break
    return l


# In[20]:


movies['cast']=movies['cast'].apply(convert3)


# In[21]:


movies.head()


# In[22]:


def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
            l.append(i['name'])
            break
    return l


# In[23]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[24]:


movies.head()


# In[25]:


movies['overview'][0]


# In[26]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[27]:


movies.head()


# In[28]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


# In[29]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[30]:


movies.head()


# In[31]:


movies['tags'] = movies['overview'] + movies['genres']+movies['keywords'] + movies['cast'] + movies['crew']


# In[32]:


movies.head()


# In[33]:


new_df=movies[['movie_id','title','tags']]


# In[34]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[35]:


new_df.head()


# In[36]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[37]:


new_df.head()


# In[38]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[39]:


def stem(text):
    y= []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[40]:


new_df['tags']=new_df['tags'].apply(stem)


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000,stop_words='english')


# In[ ]:





# In[42]:


vz=cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vz.shape


# In[44]:


cv.get_feature_names()


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity


# In[46]:


similarity=cosine_similarity(vz)


# In[47]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[48]:


similarity.shape


# In[49]:


def recommend(movie):
    movies_index =new_df[new_df['title']==movie].index[0]
    distances= similarity[movies_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# recommend('Avatar')

# In[50]:


recommend('Avatar')


# In[51]:


recommend("Batman Begins")


# In[52]:


# Using TFIDF Vectorization 


# In[53]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=1,max_features=5000,
                          ngram_range=(1,2))


# In[54]:


vz2=vectorizer.fit_transform(new_df['tags']).toarray()


# In[55]:


vz2.shape


# In[56]:


vectorizer.get_feature_names()


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity


# In[73]:


similarity=cosine_similarity(vz2)


# In[74]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[75]:


similarity.shape


# In[76]:


def recommend(movie):
    movies_index =new_df[new_df['title']==movie].index[0]
    distances= similarity[movies_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[72]:


recommend('Avatar')


# In[63]:


recommend('Liar Liar')


# In[64]:


recommend('Titanic')


# In[65]:


import pickle


# In[66]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[67]:


new_df.to_dict()


# In[69]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[77]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[81]:


def recommend(movie):
    movies_index =new_df[new_df['title']==movie].index[0]
    distances= similarity[movies_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    recommended_list=[]
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
        return recommended_movies
selected_movie_name = new_df['title'].values
recommendation = recommend(selected_movie_name)
for j in recommendation:
    print(j)


# In[ ]:




