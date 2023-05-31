

'''  code01_1장_pandas 기초.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Chapter 1: Pandas Foundations

# In[1]:


import pandas as pd
import numpy as np


# ## Introduction

# ## Dissecting the anatomy of a DataFrame

# In[2]:


pd.set_option('max_columns', 4, 'max_rows', 10)

import session_info
session_info.show()


# In[3]:


movies = pd.read_csv('data/movie.csv')
movies.head()


# ### How it works...

# ## DataFrame Attributes

# ### How to do it... {#how-to-do-it-1}

# In[4]:


movies = pd.read_csv('data/movie.csv')
columns = movies.columns
index = movies.index
data = movies.values


# In[5]:


columns


# In[6]:


index


# In[7]:


data


# In[8]:


type(index)


# In[9]:


type(columns)


# In[10]:


type(data)


# In[11]:


issubclass(pd.RangeIndex, pd.Index)


# ### How it works...

# ### There's more

# In[12]:


index.values


# In[13]:


columns.values


# ## Understanding data types

# ### How to do it... {#how-to-do-it-2}

# In[14]:


movies = pd.read_csv('data/movie.csv')


# In[15]:


movies.dtypes


# In[16]:


movies.dtypes.value_counts()


# In[17]:


movies.info()


# ### How it works...

# In[18]:


pd.Series(['Paul', np.nan, 'George']).dtype


# ### There's more...

# ### See also

# ## Selecting a Column

# ### How to do it... {#how-to-do-it-3}

# In[21]:


movies = pd.read_csv('data/movie.csv')
movies['director_name']


# In[22]:


movies.director_name


# In[23]:


movies.loc[:, 'director_name']


# In[25]:


movies.iloc[:, 1]


# In[26]:


movies['director_name'].index


# In[35]:


mdf = movies.copy()


# In[36]:


mdf.index


# In[27]:


movies['director_name'].dtype


# In[38]:


mdf.dtypes


# In[28]:


movies['director_name'].size


# In[39]:


mdf.size


# In[48]:


mdf.ndim


# In[50]:


movies['director_name'].shape


# In[51]:


mdf.shape


# In[52]:


movies['director_name'].name


# In[54]:


mdf.name


# In[45]:


mdf.index


# In[30]:


type(movies['director_name'])


# In[57]:


type(movies['director_name'])


# In[58]:


type(mdf)


# In[66]:


mdf.dtypes


# In[59]:


movies['director_name'].apply(type).


# In[65]:


movies['director_name'].apply(type).value_counts()


# In[67]:


movies['director_name'].apply(type)


# In[ ]:





# ### How it works...

# ### There's more

# ### See also

# ## Calling Series Methods

# In[68]:


s_attr_methods = set(dir(pd.Series))
len(s_attr_methods)


# In[69]:


df_attr_methods = set(dir(pd.DataFrame))
len(df_attr_methods)


# In[70]:


len(s_attr_methods & df_attr_methods)


# ### How to do it... {#how-to-do-it-4}

# In[71]:


movies = pd.read_csv('data/movie.csv')
director = movies['director_name']
fb_likes = movies['actor_1_facebook_likes']


# In[72]:


director.dtype


# In[73]:


fb_likes.dtype


# In[74]:


director.head()


# In[76]:


mdf.head()


# In[77]:


director.sample(n=5, random_state=42)


# In[78]:


mdf.sample(n = 5, random_state = 42)


# In[89]:


fb_likes.head(10)


# In[80]:


director.value_counts()


# In[82]:


fb_likes.value_counts()


# In[83]:


director.size


# In[84]:


director.shape


# In[85]:


len(director)


# In[86]:


len(mdf)


# In[87]:


director.unique()


# In[88]:


director.count()


# In[90]:


mdf.count()


# In[91]:


len(mdf) - mdf.count()


# In[92]:


len(director) - director.count()


# In[93]:


fb_likes.count()


# In[94]:


fb_likes.quantile()


# In[95]:


fb_likes.min()


# In[96]:


fb_likes.max()


# In[97]:


fb_likes.mean()


# In[98]:


fb_likes.median()


# In[99]:


fb_likes.std()


# In[100]:


fb_likes.describe()


# In[101]:


director.describe()


# In[104]:


mdf.boxplot()


# In[105]:


fb_likes.quantile(.2)


# In[106]:


fb_likes.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])


# In[115]:


fb_likes.quantile(np.linspace(start=0, stop=1, num=11))


# In[114]:


np.linspace(start=0, stop=1, num=11)


# In[116]:


director.isna()


# In[117]:


director.isna().value_counts()


# In[119]:


mdf.isna()


# In[120]:


fb_likes_filled = fb_likes.fillna(0)
fb_likes_filled.count()


# In[121]:


fb_likes_dropped = fb_likes.dropna()
fb_likes_dropped.size


# In[129]:


np.array(mdf.shape) - np.array(mdf.dropna().shape)


# In[144]:


mdfdes = mdf.describe()
mdfdes.iloc[0:1,:]


# In[145]:


type(mdfdes.iloc[0:1, :])


# In[147]:


mdfdes.iloc[0:1, :].transpose()


# In[151]:


type(mdfdes.iloc[0:1, :].transpose())


# ### How it works...

# ### There's more...

# In[154]:


director.value_counts(normalize=True)


# In[155]:


director.hasnans


# In[156]:


director.notna()


# In[157]:


director.notna().value_counts()`


# In[158]:


director.isnull()


# In[159]:


director.isnull().value_counts()


# ### See also

# ## Series Operations

# In[ ]:


5 + 9    # plus operator example. Adds 5 and 9


# ### How to do it... {#how-to-do-it-5}

# In[160]:


movies = pd.read_csv('data/movie.csv')
imdb_score = movies['imdb_score']
imdb_score


# In[ ]:


imdb_score + 1


# In[161]:


imdb_score * 2.5


# In[162]:


imdb_score // 7


# In[163]:


imdb_score > 7


# In[164]:


director = movies['director_name']
director == 'James Cameron'


# ### How it works...

# ### There's more...

# In[ ]:


imdb_score.add(1)   # imdb_score + 1


# In[ ]:


imdb_score.gt(7)   # imdb_score > 7


# ### See also

# ## Chaining Series Methods

# ### How to do it... {#how-to-do-it-6}

# In[165]:


movies = pd.read_csv('data/movie.csv')
fb_likes = movies['actor_1_facebook_likes']
director = movies['director_name']


# In[166]:


director.value_counts().head(3)


# In[168]:


fb_likes.isna().value_counts()


# In[167]:


fb_likes.isna().sum() #True가 1


# In[169]:


fb_likes.dtype


# In[170]:


(fb_likes.fillna(0)
         .astype(int)
         .head()
)


# ### How it works...

# ### There's more...

# In[171]:


(fb_likes.fillna(0)
         #.astype(int)
         #.head()
)


# In[172]:


(fb_likes.fillna(0)
         .astype(int)
         #.head()
)


# In[173]:


fb_likes.isna().mean()


# In[178]:


fb_likes.fillna(0)         # .astype(int) \
        # .head()


# In[179]:


def debug_df(df):
    print("BEFORE")
    print(df)
    print("AFTER")
    return df


# In[180]:


(fb_likes.fillna(0)
         .pipe(debug_df)
         .astype(int) 
         .head()
)


# In[181]:


intermediate = None
def get_intermediate(df):
    global intermediate
    intermediate = df
    return df


# In[182]:


res = (fb_likes.fillna(0)
         .pipe(get_intermediate)
         .astype(int) 
         .head()
)


# In[183]:


intermediate


# ## Renaming Column Names

# ### How to do it...

# In[199]:


movies = pd.read_csv('data/movie.csv')


# In[200]:


col_map = {'director_name':'Director Name', 
             'num_critic_for_reviews': 'Critical Reviews'} 


# In[201]:


movies.rename(columns=col_map).head()


# ### How it works... {#how-it-works-8}

# ### There's more {#theres-more-7}

# In[202]:


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .set_index('movie_title')
   .rename(index=idx_map, columns=col_map)
   .head(4)
)


# In[203]:


movies.set_index('movie_title').index.tolist()


# In[204]:


movies = pd.read_csv('data/movie.csv', index_col='movie_title')
ids = movies.index.tolist()
columns = movies.columns.tolist()


# In[208]:


ids[0:6]


# # rename the row and column labels with list assignments

# In[209]:


ids[0] = 'Ratava'
ids[1] = 'POC'
ids[2] = 'Ertceps'
columns[1] = 'director'
columns[-2] = 'aspect'
columns[-1] = 'fblikes'
movies.index = ids
movies.columns = columns


# In[211]:


movies.head(5)


# In[212]:


def to_clean(val):
    return val.strip().lower().replace(' ', '_')


# In[213]:


movies.rename(columns=to_clean).head(3)


# In[214]:


cols = [col.strip().lower().replace(' ', '_')
        for col in movies.columns]
movies.columns = cols
movies.head(3)


# ## Creating and Deleting columns

# ### How to do it... {#how-to-do-it-9}

# In[219]:


movies = pd.read_csv('data/movie.csv')
movies['has_seen'] = 1


# In[220]:


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .rename(index=idx_map, columns=col_map)
   .assign(has_seen=0)
)


# In[221]:


movies.head()


# In[222]:


total = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes'])


# In[223]:


total.head(5)


# In[224]:


cols = ['actor_1_facebook_likes','actor_2_facebook_likes',
    'actor_3_facebook_likes','director_facebook_likes']
sum_col = movies[cols].sum(axis='columns')
sum_col.head(5)


# In[225]:


movies.assign(total_likes=sum_col).head(5)


# In[226]:


movies.head(5)


# In[227]:


movies.assign(total_likes = movies[cols].sum(axis='columns')).head(5)


# In[228]:


def sum_likes(df):
   return df[[c for c in df.columns
              if 'like' in c]].sum(axis=1)


# In[229]:


movies.assign(total_likes=sum_likes).head(5)


# In[230]:


(movies
   .assign(total_likes=sum_col)
   ['total_likes']
   .isna()
   .sum()
)


# In[231]:


(movies
   .assign(total_likes=total)
   ['total_likes']
   .isna()
   .sum()
)


# In[232]:


(movies
   .assign(total_likes=total.fillna(0))
   ['total_likes']
   .isna()
   .sum()
)


# In[233]:


def cast_like_gt_actor_director(df):
    return df['cast_total_facebook_likes'] >=            df['total_likes']


# In[234]:


df2 = (movies
   .assign(total_likes=total,
           is_cast_likes_more = cast_like_gt_actor_director)
)


# In[235]:


df2['is_cast_likes_more'].all()


# In[236]:


df2 = df2.drop(columns='total_likes')


# In[238]:


actor_sum = (movies
   [[c for c in movies.columns if 'actor_' in c and '_likes' in c]]
   .sum(axis='columns')
)


# In[239]:


actor_sum.head(5)


# In[240]:


movies['cast_total_facebook_likes'] >= actor_sum


# In[241]:


movies['cast_total_facebook_likes'].ge(actor_sum)


# In[242]:


movies['cast_total_facebook_likes'].ge(actor_sum).all()


# In[252]:


pct_like = (actor_sum
    .div(movies['cast_total_facebook_likes'])
).mul(100)


# In[253]:


pct_like.describe()


# In[254]:


type(pct_like)


# In[261]:


pct_like.values


# In[256]:


pd.Series(pct_like.values,
    index=movies['movie_title'].values).head()


# ### How it works... {#how-it-works-9}

# ### There's more... {#theres-more-8}

# In[264]:


profit_index = movies.columns.get_loc('gross') + 1
profit_index


# In[265]:


movies.insert(loc=profit_index,
              column='profit',
              value=movies['gross'] - movies['budget'])


# In[266]:


del movies['director_name']


# ### See also


'''  code02_2장_기본 DataFrame 연산.py  '''

#!/usr/bin/env python
# coding: utf-8

import os
os.getcwd()
os.chdir('D:/pandas_cookbook')

# # Chapter 2: Essential DataFrame Operations

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# In[4]:


dir()


# ## Introduction

# ## Selecting Multiple DataFrame Columns

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
movie_actor_director = movies[['actor_1_name', 
                               'actor_2_name',
                               'actor_3_name', 
                               'director_name']]
movie_actor_director.head()

# movie_actor_director.shape

# In[ ]:


type(movies[['director_name']])


# In[ ]:


type(movies['director_name'])


# In[ ]:


type(movies.loc[:, ['director_name']])


# In[ ]:


t3ype(movies.loc[:, 'director_name'])


# ### How it works\...

# ### There\'s more\...

# In[ ]:


cols = ['actor_1_name', 
        'actor_2_name',
        'actor_3_name', 
        'director_name']

movie_actor_director = movies[cols]


# In[ ]:


# key error 발생 사례

movies['actor_1_name', 
       'actor_2_name',
       'actor_3_name', 
       'director_name']


# ## Selecting Columns with Methods

# ### How it works\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )

movies = movies.rename(columns=shorten)

movies.dtypes.value_counts()


# In[ ]:


movies.select_dtypes(include='int').head()

movies.select_dtypes(include='int64').head()


# In[ ]:


movies.select_dtypes(include='number').head()

#int랑 float이 모두 추출됨


# In[ ]:


movies.select_dtypes(include=['int64', 'object']).head()


# In[ ]:


movies.select_dtypes(exclude='float').head()


# In[ ]:


movies.filter(like='fb').head()


# In[ ]:


cols = ['actor_1_name', 'actor_2_name','actor_3_name', 'director_name']

movies.filter(items=cols).head()


# In[ ]:


movies.filter(regex = r'\d').head()


# ### How it works\...

# ### There\'s more\...

# ### See also

# ## Ordering Column Names

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )
movies = movies.rename(columns = shorten)


# In[ ]:


movies.columns


# In[ ]:


cat_core = ['movie_title', 
            'title_year',
            'content_rating', 
            'genres']

cat_people = ['director_name', 
              'actor_1_name',
              'actor_2_name', 
              'actor_3_name']

cat_other = ['color', 
             'country', 
             'language',
             'plot_keywords', 
             'movie_imdb_link']

cont_fb = ['director_fb', 
           'actor_1_fb',
           'actor_2_fb', 
           'actor_3_fb',
           'cast_total_fb', 
           'movie_fb']

cont_finance = ['budget', 'gross']

cont_num_reviews = ['num_voted_users', 
                    'num_user',
                    'num_critic']

cont_other = ['imdb_score', 
              'duration',
               'aspect_ratio', 
               'facenumber_in_poster']


# In[ ]:


new_col_order = cat_core + cat_people + \
                cat_other + \
                cont_fb + \
                cont_finance + \
                cont_num_reviews + \
                cont_other

print(new_col_order)
                
set(movies.columns) == set(new_col_order)


# In[ ]:


movies[new_col_order].head()

type(movies[new_col_order])
type(new_col_order)


# ### How it works\...

# ### There\'s more\...

# ### See also

# ## Summarizing a DataFrame

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
movies.shape


# In[ ]:


movies.size


# In[ ]:


movies.ndim


# In[ ]:


len(movies)


# In[ ]:


movies.count()


# In[ ]:


movies.min()


# In[ ]:


movies.describe().T
type(movies.describe().T)

# print(movies.describe().T)
# In[ ]:


movies.describe(percentiles=[.01, .3, .99]).T


# ### How it works\...

# ### There\'s more\...

# In[ ]:


movies.min(skipna = False)


# ## Chaining DataFrame Methods

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )
movies = movies.rename(columns=shorten)
movies.isnull().head()


# In[ ]:


(movies
   .isnull()
   .sum()
  # .head()
)

# In[ ]:

# 총 결측치 개수 확인
movies.isnull().sum().sum()


# In[ ]:


movies.isnull().any().any()

# movies.isnull().any().all()

# ### How it works\...

# In[ ]:


movies.isnull().dtypes.value_counts()


# ### There\'s more\...

# In[ ]:

# 기본 집계 메서드는 결측치가 있을 때, 아무것도 반환하지 않음
movies[['color', 'movie_title', 'color']].max()


# In[ ]:

# 각 열에 대해 무언가를 반환하게 하려면 결측치를 채워야 함
with pd.option_context('max_colwidth', 20):
    movies.select_dtypes(['object']).fillna('').max()


# In[ ]:


with pd.option_context('max_colwidth', 20):
    (movies
        .select_dtypes(['object'])
        .fillna('')
        .max()
    )

# ### See also

# ## DataFrame Operations

# In[ ]:


colleges = pd.read_csv('data/college.csv')
colleges + 5


# In[ ]:


colleges = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = colleges.filter(like='UGDS_')
college_ugds.head()

college_ugds.index

# In[ ]:


name = 'Northwest-Shoals Community College'
college_ugds.loc[name]


# In[ ]:

# bankers rounding -> college_ugds.loc[name].round(2)


# In[ ]:


(college_ugds.loc[name] + .0001).round(2)


# In[ ]:


college_ugds + .00501


# In[ ]:


(college_ugds + .00501) // .01


# In[ ]:


college_ugds_op_round = (college_ugds + .00501) // .01 / 100
college_ugds_op_round.head()


# In[ ]:


college_ugds_round = (college_ugds + .00001).round(2)
college_ugds_round


# In[ ]:


college_ugds_op_round.equals(college_ugds_round)


# ### How it works\...

# In[ ]:

# 부동소수점
.045 + .005


# ### There\'s more\...

# In[ ]:


college2 = (college_ugds
    .add(.00501) 
    .floordiv(.01) 
    .div(100)
)
college2.equals(college_ugds_op_round)


# ### See also

# ## Comparing Missing Values

# In[ ]:

# 결측치 비교
np.nan == np.nan


# In[ ]:


None == None


# In[ ]:


np.nan > 5


# In[ ]:


5 > np.nan


# In[ ]:


np.nan != 5


# ### Getting ready

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')


# In[ ]:


college_ugds == .0019


# In[ ]:


college_self_compare = college_ugds == college_ugds
college_self_compare.head()


# In[ ]:


college_self_compare.all()


# In[ ]:


(college_ugds == np.nan).sum()


# In[ ]:


college_ugds.isnull().sum()


# In[ ]:


college_ugds.equals(college_ugds)


# ### How it works\...

# ### There\'s more\...

# In[ ]:


college_ugds.eq(.0019)    # same as college_ugds == .0019


# In[ ]:


from pandas.testing import assert_frame_equal
assert_frame_equal(college_ugds, college_ugds) is None


# ## Transposing the direction of a DataFrame operation

# ### How to do it\...

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


# In[ ]:

college.shape
college_ugds.count()


# In[ ]:


college_ugds.count(axis='columns').head()


# In[ ]:


college_ugds.sum(axis='columns').head()


# In[ ]:


college_ugds.median(axis='index')


# ### How it works\...

# ### There\'s more\...

# In[ ]:


college_ugds_cumsum = college_ugds.cumsum(axis=1)
college_ugds_cumsum.head()

# ### See also

# ## Determining college campus diversity

# In[ ]:


pd.read_csv('data/college_diversity.csv', index_col='School')


# ### How to do it\...

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')


# In[ ]:


(college_ugds.isnull()
   .sum(axis='columns')
   .sort_values(ascending=False)
   .head()
)


# In[ ]:


college_ugds = college_ugds.dropna(how='all')
college_ugds.isnull().sum()


# In[ ]:


college_ugds.ge(.15)


# In[ ]:


diversity_metric = college_ugds.ge(.15).sum(axis='columns')
diversity_metric.head()


# In[ ]:


diversity_metric.value_counts()


# In[ ]:


diversity_metric.sort_values(ascending=False).head()


# In[ ]:


college_ugds.loc[['Regency Beauty Institute-Austin',
                   'Central Texas Beauty College-Temple']]


# In[ ]:


us_news_top = ['Rutgers University-Newark',
                  'Andrews University',
                  'Stanford University',
                  'University of Houston',
                  'University of Nevada-Las Vegas']
diversity_metric.loc[us_news_top]


# ### How it works\...

# ### There\'s more\...

# In[ ]:


(college_ugds
   .max(axis=1)
   .sort_values(ascending=False)
   .head(10)
)


# In[ ]:


(college_ugds > .01).all(axis=1).any()



# ### See also


'''  code03_3장_DataFrame 생성과 유지.py  '''

#!/usr/bin/env python
# coding: utf-8

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")


# In[ ]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ### How to do it\...

# In[ ]:


fname = ['Paul', 'John', 'Richard', 'George']
lname = ['McCartney', 'Lennon', 'Starkey', 'Harrison']
birth = [1942, 1940, 1940, 1943]


# In[ ]:


people = {'first': fname, 'last': lname, 'birth': birth}


# In[ ]:


beatles = pd.DataFrame(people)
beatles


# ### How it works\...

# In[ ]:


beatles.index


# In[ ]:


pd.DataFrame(people, index=['a', 'b', 'c', 'd'])


# ### There\'s More

# In[ ]:


pd.DataFrame(
[{"first":"Paul", "last":"McCartney", "birth":1942},
 {"first":"John", "last":"Lennon", "birth":1940},
 {"first":"Richard", "last":"Starkey", "birth":1940},
 {"first":"George", "last":"Harrison", "birth":1943}])


# In[ ]:

pd.DataFrame(
[{"first":"Paul","last":"McCartney", "birth":1942},
 {"first":"John","last":"Lennon", "birth":1940},
 {"first":"Richard","last":"Starkey", "birth":1940},
 {"first":"George","last":"Harrison", "birth":1943}],
 columns=['last', 'first', 'birth'])


# ### How to do it\...

# In[ ]:


beatles


# In[ ]:


from io import StringIO
fout = StringIO()
beatles.to_csv(fout)  # use a filename instead of fout
# beatles.to_csv("test.csv") 

# In[ ]:


print(fout.getvalue())


# ### There\'s More

# In[ ]:


_ = fout.seek(0)
pd.read_csv(fout)


# In[ ]:


_ = fout.seek(0)
pd.read_csv(fout, index_col=0)


# In[ ]:


fout = StringIO()
beatles.to_csv(fout, index=False) 
print(fout.getvalue())


# ### How to do it\...

# In[ ]:

os.listdir()

diamonds = pd.read_csv('data/diamonds.csv', nrows=1000)
diamonds


# In[ ]:


diamonds.info()


# In[ ]:


diamonds2 = pd.read_csv('data/diamonds.csv', 
                        nrows=1000,
                        dtype={'carat': np.float32, 
                               'depth': np.float32,
                               'table': np.float32, 
                               'x': np.float32,
                               'y': np.float32, 
                               'z': np.float32,
                               'price': np.int16})


# In[ ]:


diamonds2.info()


# In[ ]:


diamonds.describe()


# In[ ]:


diamonds2.describe()


# In[ ]:


diamonds2.cut.value_counts()


# In[ ]:


diamonds2.color.value_counts()


# In[ ]:


diamonds2.clarity.value_counts()


# In[ ]:


diamonds3 = pd.read_csv('data/diamonds.csv', 
                        nrows=1000,
                        dtype={'carat': np.float32, 
                               'depth': np.float32,
                               'table': np.float32, 
                                  'x': np.float32,
                                  'y': np.float32, 
                                  'z': np.float32,
                                  'price': np.int16,
                                  'cut': 'category', 
                                  'color': 'category',
                                  'clarity': 'category'})


# In[ ]:


diamonds3.info()


# In[ ]:


np.iinfo(np.int8)
np.iinfo(np.int16)
np.iinfo(np.int32)
np.iinfo(np.int64)


# In[ ]:


np.finfo(np.float16)
np.finfo(np.float32)
np.finfo(np.float64)


# In[ ]:

# col_selection
cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']

diamonds4 = pd.read_csv('data/diamonds.csv', 
                        nrows=1000,
                        dtype={'carat': np.float32, 
                               'depth': np.float32,
                               'table': np.float32, 
                               'price': np.int16,
                               'cut': 'category', 
                               'color': 'category',
                               'clarity': 'category'},
                        usecols = cols)

diamonds4.dtypes

# In[ ]:


diamonds4.info()


# In[ ]:

# Chunk 이용
cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']

diamonds_iter = pd.read_csv('data/diamonds.csv', 
                            nrows=1000,
                            dtype={'carat': np.float32, 
                                   'depth': np.float32,
                                   'table': np.float32, 
                                   'price': np.int16,
                                   'cut': 'category', 
                                   'color': 'category',
                                   'clarity': 'category'},
                            usecols = cols,
                            chunksize = 200)


# In[ ]:


def process(df):
    return f'processed {df.size} items'


# In[ ]:


for chunk in diamonds_iter:
    process(chunk)


# ### How it works\...

# ### There\'s more \...

# In[ ]:

# Return the memory usage of each column in bytes
diamonds.price.memory_usage()

# In[ ]:


diamonds.price.memory_usage(index=False)


# In[ ]:


diamonds.cut.memory_usage()


# In[ ]:


diamonds.cut.memory_usage(deep=True)



# In[ ]:

# Feather 형식과 같이 형식을 추적하는 이진 형식으로 저장할 수 있음

# read_feather() reads a Feather file as a pandas.DataFrame. 
# read_table() reads a Feather file as a Table. Internally, 
# read_feather() simply calls read_table() and the result is converted to pandas:

import pyarrow.feather as feather
diamonds4.to_feather('tmp/d.arr')
diamonds5 = pd.read_feather('tmp/d.arr')

diamonds5.info() # 17.7 MB

# os.mkdir('tmp')
# os.listdir()


# In[ ]:


diamonds4.to_parquet('tmp/d.pqt')
diamonds6 = pd.read_parquet('tmp/d.pqt')
diamonds6.info()


# ### How to do it\...

# In[ ]:


beatles.to_excel('tmp/beat.xls')


# In[ ]:


beatles.to_excel('tmp/beat.xlsx')


# In[ ]:


beat2 = pd.read_excel('tmp/beat.xls')
beat2


# In[ ]:


beat2 = pd.read_excel('tmp/beat.xls', index_col=0)
beat2


# In[ ]:


beat2.dtypes


# ### How it works\...

# ### There\'s more\...

# In[ ]:

# 판다스 이용 스프레드 시트 내 특정 시트 작성

xl_writer = pd.ExcelWriter('tmp/beat.xlsx')
beatles.to_excel(xl_writer, sheet_name='All')
beatles[beatles.birth < 1941].to_excel(xl_writer, sheet_name = '1940')
xl_writer.save()



# ### How to do it\...

# In[ ]:


autos = pd.read_csv('data/vehicles.csv.zip')
autos


# In[ ]:


autos.modifiedOn.dtype


# In[ ]:


autos.modifiedOn


# In[ ]:


pd.to_datetime(autos.modifiedOn)  # doctest: +SKIP


# In[ ]:


autos = pd.read_csv('data/vehicles.csv.zip',
    parse_dates=['modifiedOn'])  # doctest: +SKIP
autos.modifiedOn


# In[ ]:


import zipfile


# In[ ]:


with zipfile.ZipFile('data/kaggle-survey-2018.zip') as z:
    print('\n'.join(z.namelist()))
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    kag_questions = kag.iloc[0]
    survey = kag.iloc[1:]


# In[ ]:


print(survey.head(2).T)


# ### How it works\...

# ### There\'s more\...

# ### How to do it\...

# In[ ]:


import sqlite3

con = sqlite3.connect('data/beat.db')

with con:
    cur = con.cursor()
    cur.execute("""DROP TABLE Band""")
    cur.execute("""CREATE TABLE Band(id INTEGER PRIMARY KEY,
        fname TEXT, lname TEXT, birthyear INT)""")
    cur.execute("""INSERT INTO Band VALUES(
        0, 'Paul', 'McCartney', 1942)""")
    cur.execute("""INSERT INTO Band VALUES(
        1, 'John', 'Lennon', 1940)""")
    _ = con.commit()


# In[ ]:


import sqlalchemy as sa

engine = sa.create_engine(
  'sqlite:///data/beat.db', echo=True)

sa_connection = engine.connect()


# In[ ]:


beat = pd.read_sql('Band', sa_connection, index_col='id')
beat


# In[ ]:


sql = '''SELECT fname, birthyear from Band'''
fnames = pd.read_sql(sql, con)
fnames


# ### How it work\'s\...

# In[ ]:


import json
encoded = json.dumps(people)
encoded


# In[ ]:


json.loads(encoded)


# ### How to do it\...

# In[ ]:


beatles = pd.read_json(encoded)
beatles


# In[ ]:


records = beatles.to_json(orient='records')
records


# In[ ]:


pd.read_json(records, orient='records')


# In[ ]:


split = beatles.to_json(orient='split')
split


# In[ ]:


pd.read_json(split, orient='split')


# In[ ]:


index = beatles.to_json(orient='index')
index


# In[ ]:


pd.read_json(index, orient='index')


# In[ ]:


values = beatles.to_json(orient='values')
values


# In[ ]:


pd.read_json(values, orient='values')


# In[ ]:


(pd.read_json(values, orient='values')
   .rename(columns=dict(enumerate(['first', 'last', 'birth'])))
)


# In[ ]:


table = beatles.to_json(orient='table')
table


# In[ ]:


pd.read_json(table, orient='table')


# ### How it works\...

# ### There\'s more\...

# In[ ]:

# 웹 서비스 작업 중이고 JSON에 데이터를 추가해야 하는 상황
# .to_dict 메서드 이용 딕셔너리 생성
output = beat.to_dict()
output


# In[ ]:


output['version'] = '0.4.1'
json.dumps(output)


# ### How to do it\...

# In[ ]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url)
len(dfs)


# In[ ]:


dfs[0]


# In[ ]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—')
len(dfs)


# In[ ]:


dfs[0].columns


# In[ ]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—',
    header=[0,1])
len(dfs)


# In[ ]:


dfs[0]


# In[ ]:


dfs[0].columns


# In[ ]:


df = dfs[0]
df.columns = ['Title', 'Release', 'UK', 'AUS', 'CAN', 'FRA', 'GER',
    'NOR', 'US', 'Certifications']
df


# In[ ]:


res = (df
  .pipe(lambda df_: df_[~df_.Title.str.startswith('Released')])
  .iloc[:-1]
  .assign(release_date=lambda df_: pd.to_datetime(
             df_.Release.str.extract(r'Released: (.*) Label')
               [0]
               .str.replace(r'\[E\]', '')
          ),
          label=lambda df_:df_.Release.str.extract(r'Label: (.*)')
         )
   .loc[:, ['Title', 'UK', 'AUS', 'CAN', 'FRA', 'GER', 'NOR',
            'US', 'release_date', 'label']]
)
res


# ### How it works\...

# ### There is more\...

# In[ ]:


url = 'https://github.com/mattharrison/datasets/blob/master/data/anscombes.csv'
dfs = pd.read_html(url, attrs={'class': 'csv-data'})
len(dfs)


# In[ ]:


dfs[0]


# In[ ]:






'''  code04_4장_데이터 분석 시작.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Beginning Data Analysis 

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name) # name object 삭제

# In[ ]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Developing a data analysis routine

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv')
college.sample(random_state = 42)


# In[ ]:


college.shape


# In[ ]:


college.info()


# In[ ]:


college.describe(include=[np.number]).T

college.shape[0] - college.count()


# In[ ]:

college.describe(include = [np.object]).T
college.describe(include = [pd.Categorical]).T
college.describe(include=[np.object, pd.Categorical]).T


# ### How it works...

# ### There's more...

# In[ ]:


college.describe(include=[np.number],
   percentiles=[.01, .05, .10, .25, .5,
                .75, .9, .95, .99]).T



# ## Data dictionaries

# In[ ]:


pd.read_csv('data/college_data_dictionary.csv')


# ## Reducing memory by changing data types

# ### How to do it...

# In[ ]:
college = pd.read_csv('data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER',
                  'INSTNM', 'STABBR']
col2 = college.loc[:, different_cols]
col2.head()


# In[ ]:


col2.dtypes


# In[ ]:


original_mem = col2.memory_usage(deep=True)
original_mem

col2.info()

# In[ ]:


col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)    


# In[ ]:


col2.dtypes


# In[ ]:


college[different_cols].memory_usage(deep=True)
col2.memory_usage(deep=True)

# In[ ]:


col2.select_dtypes(include = ['object']).nunique()


# In[ ]:


col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes


# In[ ]:


new_mem = col2.memory_usage(deep = True)
new_mem


# In[ ]:


new_mem / original_mem


# ### How it works...

# ### There's more...

# In[ ]:


college.loc[0, 'CURROPER'] = 10000000
college.loc[0, 'INSTNM'] = college.loc[0, 'INSTNM'] + 'a'
college[['CURROPER', 'INSTNM']].memory_usage(deep=True)


# In[ ]:


college['MENONLY'].dtype


# In[ ]:


college['MENONLY'].astype(np.int8)


# In[ ]:


college.assign(MENONLY=college['MENONLY'].astype('float16'),
    RELAFFIL=college['RELAFFIL'].astype('int8'))


# In[ ]:


college.index = pd.Int64Index(college.index)
college.index.memory_usage() # previously was just 80


# ## Selecting the smallest of the largest

# ### How to do it...

# In[ ]:

# 이 예제에서는 .nlargest와 .nsmallest 와 같은 편리한 메서드를 사용해 
# 가장 저렴한 예산으로 만들어진 영화 찾기
movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()


# In[ ]:


movie2.nlargest(100, 'imdb_score').head()


# In[ ]:


(movie2
  .nlargest(100, 'imdb_score')
  .nsmallest(5, 'budget')
)


# ### How it works...

# ### There's more...

# ## Selecting the largest of each group by sorting

# ### How to do it...

# In[ ]:


movie = pd.read_csv('data/movie.csv')
movie[['movie_title', 'title_year', 'imdb_score']]


# In[ ]:

# 내림차순
(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values('title_year', ascending=False)
)


# In[ ]:


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values(['title_year','imdb_score'],
               ascending=False)
)


# In[ ]:


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values(['title_year','imdb_score'],
               ascending=False)
  .drop_duplicates(subset='title_year')
)




# ### How it works...

# ## There's more...

# In[ ]:


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .groupby('title_year', as_index=False)
  .apply(lambda df: 
             df.sort_values('imdb_score',
                            ascending=False)
             .head(1))
  .sort_values('title_year', ascending=False)
)


# In[ ]:


(movie
  [['movie_title', 'title_year', 'content_rating', 'budget']]
   .sort_values(['title_year','content_rating', 'budget'],
                ascending = [False, False, True])
   .drop_duplicates(subset = ['title_year', 'content_rating'])
)


# ## Replicating nlargest with sort_values

# ### How to do it...

# In[ ]:


movie = pd.read_csv('data/movie.csv')

(movie
   [['movie_title', 'imdb_score', 'budget']]
   .nlargest(100, 'imdb_score') 
   .nsmallest(5, 'budget')
)


# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False)
   .head(100)
)


# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False)
   .head(100) 
   .sort_values('budget')
   .head(5)
)


# ### How it works...

# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .nlargest(100, 'imdb_score')
   .tail()
)


# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False) 
   .head(100)
   .tail()
)


# ## Calculating a trailing stop order price

# ### How to do it...

# In[ ]:

# 설치 명령어
# conda install -c conda-forge requests-cache

import datetime
import pandas_datareader.data as web
import requests_cache

session = (requests_cache.CachedSession(cache_name='cache', 
                                        backend='sqlite', 
                                        expire_after = datetime.timedelta(days=90))
            )

# !pip install requests_cache
# In[ ]:


tsla = web.DataReader('tsla', 
                      data_source = 'yahoo',
                      start = '2017-1-1', 
                      session = session)

# tsla = pd.read_csv("data/TSLA.csv")

tsla.head(8)


# In[ ]:


tsla_close = tsla['Close']


# In[ ]:


tsla_cummax = tsla_close.cummax()
tsla_cummax.head()


# In[ ]:


(tsla
  ['Close']
  .cummax() # 현재까지 종가 중에 최대값 추적
  .mul(.9)
  .head()
)


# ### How it works...

# ### There's more...


'''  code05_5장_탐색적 데이터 분석.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis


import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name) # name object 삭제



# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 10, 'max_rows', 10, 'max_colwidth', 50)


# ## Introduction

# ## Summary Statistics

# ### How to do it...

# In[2]:

# fueleconomy.gov 데이터 셋 이용 
fueleco = pd.read_csv('data/vehicles.csv.zip')
fueleco


# In[3]:


fueleco.mean() # doctest: +SKIP


# In[4]:


fueleco.std() # doctest: +SKIP


# In[5]:


fueleco.quantile([0, .25, .5, .75, 1]) # doctest: +SKIP


# In[6]:


fueleco.describe()  # doctest: +SKIP

# fueleco.isna().sum()

type(fueleco.isna().sum() >= 1)

tmp = fueleco.isna().sum() >= 1

for idx, value in enumerate(tmp):
    if value == True:
        print(tmp.index[idx])
        



# In[7]:


fueleco.describe(include=object).T  # doctest: +SKIP


# ### How it works...

# ### There's more...

# In[8]:


fueleco.describe().T    # doctest: +SKIP


# ## Column Types

# ### How to do it...

# In[9]:


fueleco.dtypes


# In[10]:


fueleco.dtypes.value_counts()


# ### How it works...

# ### There's more...

# In[11]:


fueleco.select_dtypes('int64').describe().T


# In[12]:


np.iinfo(np.int8)


# In[13]:


np.iinfo(np.int16)


# In[14]:


fueleco[['city08', 'comb08']].info()


# In[15]:


(fueleco
  [['city08', 'comb08']]
  .assign(city08=fueleco.city08.astype(np.int16),
          comb08=fueleco.comb08.astype(np.int16))
  .info()
)


# In[16]:


fueleco.make.nunique()


# In[17]:


fueleco.model.nunique()


# In[18]:

# Series에는 info 메서드가 없으므로

fueleco[['make']].info()
type(fueleco[['make']])


# In[19]:


(fueleco
    [['make']]
    .assign(make=fueleco.make.astype('category'))
    .info()
)


# In[20]:


fueleco[['model']].info()


# In[21]:


(fueleco
    [['model']]
    .assign(model=fueleco.model.astype('category'))
    .info()
)


# ## Categorical Data

# ### How to do it...

# In[22]:


fueleco.select_dtypes(object).columns


# In[23]:


fueleco.drive.nunique()


# In[24]:


fueleco.drive.sample(5, random_state = 42)


# In[25]:


fueleco.drive.isna().sum()


# In[26]:

# 결측치의 % 확인
fueleco.drive.isna().mean() * 100


# In[27]:


fueleco.drive.value_counts()


# In[28]:


top_n = fueleco.make.value_counts().index[:6]

(fueleco
   .assign(make=fueleco.make.where(
              fueleco.make.isin(top_n),
              'Other'))
   .make
   .value_counts()
)


# In[29]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
top_n = fueleco.make.value_counts().index[:6]
(fueleco     # doctest: +SKIP
   .assign(make=fueleco.make.where(
              fueleco.make.isin(top_n),
              'Other'))
   .make
   .value_counts()
   .plot.bar(ax=ax)
)
fig.savefig('tmp/c5-catpan.png', dpi=300)     # doctest: +SKIP


# In[30]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(10, 8))
top_n = fueleco.make.value_counts().index[:6]
sns.countplot(y='make',     # doctest: +SKIP
  data= (fueleco
   .assign(make=fueleco.make.where(
              fueleco.make.isin(top_n),
              'Other'))
  )
)
fig.savefig('tmp/c5-catsns.png', dpi=300)    # doctest: +SKIP


# ### How it works...

# In[31]:


fueleco[fueleco.drive.isna()]


# In[32]:


fueleco.drive.value_counts(dropna = False)


# ### There's more...

# In[33]:


fueleco.rangeA.value_counts()


# In[34]:

# 정규표현식 이용 특수문자 개수 찾
(fueleco
 .rangeA
 .str.extract(r'([^0-9.])')
 .dropna()
 .apply(lambda row: ''.join(row), axis=1)
 .value_counts()
)


# In[35]:


set(fueleco.rangeA.apply(type))
fueleco.rangeA.apply(type).value_counts()

# In[36]:


fueleco.rangeA.isna().sum()


# In[37]:


(fueleco
  .rangeA
  .fillna('0')
  .str.replace('-', '/')
  .str.split('/', expand=True)
  .astype(float)
  .mean(axis=1)
)


# In[38]:


(fueleco
  .rangeA
  .fillna('0')
  .str.replace('-', '/')
  .str.split('/', expand=True)
  .astype(float)
  .mean(axis=1)
  .pipe(lambda ser_: pd.cut(ser_, 10))
  .value_counts()
)


# In[39]:


(fueleco
  .rangeA
  .fillna('0')
  .str.replace('-', '/')
  .str.split('/', expand=True)
  .astype(float)
  .mean(axis=1)
  .pipe(lambda ser_: pd.qcut(ser_, 10))
  .value_counts()
)


# In[40]:


(fueleco
  .city08
  .pipe(lambda ser: pd.qcut(ser, q=10))
  .value_counts()
)


# ## Continuous Data

# ### How to do it...

# In[41]:


fueleco.select_dtypes('number')


# In[42]:


fueleco.city08.sample(5, random_state=42)


# In[43]:


fueleco.city08.isna().sum()


# In[44]:


fueleco.city08.isna().mean() * 100


# In[45]:


fueleco.city08.describe()


# In[46]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fueleco.city08.hist(ax=ax)
fig.savefig('tmp/c5-conthistpan.png', dpi=300)     # doctest: +SKIP


# In[47]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fueleco.city08.hist(ax=ax, bins=30)
fig.savefig('tmp/c5-conthistpanbins.png', dpi=300)     # doctest: +SKIP


# In[48]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.distplot(fueleco.city08, rug=True, ax=ax)
fig.savefig('tmp/c5-conthistsns.png', dpi=300)  -   # doctest: +SKIP


# ### How it works...

# ### There's more...

# In[49]:


fig, axs = plt.subplots(nrows=3, figsize=(10, 8))
sns.boxplot(fueleco.city08, ax=axs[0])
sns.violinplot(fueleco.city08, ax=axs[1])
sns.boxenplot(fueleco.city08, ax=axs[2])
fig.savefig('tmp/c5-contothersns.png', dpi=300)     


# In[50]:


from scipy import stats
stats.kstest(fueleco.city08, cdf='norm')


# In[51]:

# QQ-plot

from scipy import stats
fig, ax = plt.subplots(figsize=(10, 8))
stats.probplot(fueleco.city08, plot=ax)
fig.savefig('tmp/c5-conprob.png', dpi=300)    


# ## Comparing Continuous Values across Categories

# ### How to do it...

# In[52]:


mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
fueleco[mask].groupby('make').city08.agg(['mean', 'std'])


# In[53]:

# 여러 범주의 연속형 변수 비교

g = sns.catplot(x='make', 
                y='city08', 
                data=fueleco[mask], 
                kind='box')

g.ax.figure.savefig('tmp/c5-catbox.png', dpi=300)     


# ### How it works...

# ### There's more...

# In[54]:


mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
(fueleco
  [mask]
  .groupby('make')
  .city08
  .count()
)


# In[55]:


g = sns.catplot(x='make', y='city08', 
                data=fueleco[mask], 
                kind='box')

sns.swarmplot(x='make', y='city08',    # doctest: +SKIP
              data=fueleco[mask], 
              color='k', 
              size=1, 
              ax=g.ax)

g.ax.figure.savefig('tmp/c5-catbox2.png', dpi=300)    # doctest: +SKIP  


# In[56]:


g = sns.catplot(x='make', y='city08', 
                data=fueleco[mask], 
                kind='box',
                col='year', 
                col_order=[2012, 2014, 2016, 2018],
                col_wrap=2)
g.axes[0].figure.savefig('tmp/c5-catboxcol.png', dpi=300)    # doctest: +SKIP  


# In[57]:


g = sns.catplot(x='make', y='city08', # doctest: +SKIP  
                data=fueleco[mask], 
                kind='box',
                hue='year', 
                hue_order=[2012, 2014, 2016, 2018])
g.ax.figure.savefig('tmp/c5-catboxhue.png', dpi=300)    # doctest: +SKIP  


# In[58]:


mask = fueleco.make.isin(['Ford', 'Honda', 'Tesla', 'BMW'])
grd = (fueleco
  [mask]
  .groupby('make')
  .city08
  .agg(['mean', 'std'])
  .style.background_gradient(cmap='RdBu', axis=0)
)


# ## Comparing Two Continuous Columns

# ### How to do it...

# In[59]:

# 공분산 비교

fueleco.city08.cov(fueleco.highway08)


# In[60]:


fueleco.city08.cov(fueleco.comb08)


# In[61]:


fueleco.city08.cov(fueleco.cylinders)


# In[62]:


fueleco.city08.corr(fueleco.highway08)


# In[63]:


fueleco.city08.corr(fueleco.cylinders)


# In[64]:


import seaborn as sns

fig, ax = plt.subplots(figsize=(8,8))
corr = fueleco[['city08', 'highway08', 'cylinders']].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, 
            mask=mask,
            fmt='.2f', 
            annot=True, 
            ax=ax, 
            cmap='RdBu', 
            vmin=-1, 
            vmax=1,
            square=True)

fig.savefig('tmp/c5-heatmap.png', dpi=300, bbox_inches='tight')


# In[65]:

# scatter plot

fig, ax = plt.subplots(figsize=(8,8))
fueleco.plot.scatter(x='city08', y='highway08', alpha=1, ax=ax)
fig.savefig('tmp/c5-scatpan.png', dpi=300, bbox_inches='tight')


# In[66]:


fig, ax = plt.subplots(figsize=(8,8))
fueleco.plot.scatter(x='city08', y='cylinders', alpha=.1, ax=ax)
fig.savefig('tmp/c5-scatpan-cyl.png', dpi=300, bbox_inches='tight')


# In[67]:


fueleco.cylinders.isna().sum()


# In[68]:


fig, ax = plt.subplots(figsize=(8,8))
(fueleco
 .assign( cylinders = fueleco.cylinders.fillna(0))
 .plot.scatter(x='city08', y='cylinders', alpha=.1, ax=ax))

fig.savefig('tmp/c5-scatpan-cyl0.png', dpi=300, bbox_inches='tight')


# In[69]:


res = sns.lmplot(x='city08', y='highway08', data=fueleco) 
res.fig.savefig('tmp/c5-lmplot.png', dpi=300, bbox_inches='tight')


# ### How it works...

# In[70]:


fueleco.city08.corr(fueleco.highway08*2)


# In[71]:


fueleco.city08.cov(fueleco.highway08*2)


# ### There's more...

# In[72]:


res = sns.relplot(x='city08', y='highway08',
   data=fueleco.assign(cylinders=fueleco.cylinders.fillna(0)),
                       hue='year', 
                       size='barrels08', 
                       alpha=.5, 
                       height=8)
res.fig.savefig('tmp/c5-relplot2.png', dpi=300, bbox_inches='tight')


# In[73]:


res = sns.relplot(x='city08', y='highway08',
                  data=fueleco.assign(cylinders=fueleco.cylinders.fillna(0)),
                  hue='year', 
                  size='barrels08', 
                  alpha=.5, 
                  height=8,
                  col='make', 
                  col_order=['Ford', 'Tesla'])

res.fig.savefig('tmp/c5-relplot3.png', dpi=300, bbox_inches='tight')


# In[74]:


fueleco.city08.corr(fueleco.barrels08, method='spearman')


# ## Comparing Categorical and Categorical Values

# ### How to do it...

# In[75]:
# 범주 값과 범주 값 비교

def generalize(ser, match_name, default):
    seen = None
    for match, name in match_name:
        mask = ser.str.contains(match)
        if seen is None:
            seen = mask
        else:
            seen |= mask
        ser = ser.where(~mask, name)
    ser = ser.where(seen, default)
    return ser


# In[76]:


makes = ['Ford', 'Tesla', 'BMW', 'Toyota']
data = (fueleco
   [fueleco.make.isin(makes)]
   .assign(SClass=lambda df_: generalize(df_.VClass,
    [('Seaters', 'Car'), ('Car', 'Car'), ('Utility', 'SUV'),
     ('Truck', 'Truck'), ('Van', 'Van'), ('van', 'Van'),
     ('Wagon', 'Wagon')], 'other'))
)


# In[77]:


data.groupby(['make', 'SClass']).size().unstack()


# In[78]:


pd.crosstab(data.make, data.SClass)


# In[79]:


pd.crosstab([data.year, data.make], [data.SClass, data.VClass])


# In[80]:


import scipy.stats as ss
import numpy as np
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[81]:


cramers_v(data.make, data.SClass)

data.make.corr(data.SClass, cramers_v)


# In[82]:


fig, ax = plt.subplots(figsize=(10,8))
(data
 .pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
 .plot.bar(ax=ax)
)
fig.savefig('tmp/c5-bar.png', dpi=300, bbox_inches='tight')


# In[83]:


res = sns.catplot(kind='count',
   x='make', hue='SClass', data=data)
res.fig.savefig('tmp/c5-barsns.png', dpi=300, bbox_inches='tight')


# In[84]:


fig, ax = plt.subplots(figsize=(10,8))
(data
 .pipe(lambda df_: pd.crosstab(df_.make, df_.SClass))
 .pipe(lambda df_: df_.div(df_.sum(axis=1), axis=0))
 .plot.bar(stacked=True, ax=ax)
)
fig.savefig('tmp/c5-barstacked.png', dpi=300, bbox_inches='tight')


# ### How it works...

# In[85]:


cramers_v(data.make, data.trany)


# In[86]:


cramers_v(data.make, data.model)


# ## Using the Pandas Profiling Library

# ### How to do it...

# In[87]:


import pandas_profiling as pp
pp.ProfileReport(fueleco)
report = pp.ProfileReport(fueleco)
report.to_file('tmp/fuel.html')




'''  code06_6장_데이터의 부분집합 선택.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Selecting Subsets of Data 

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name) # name object 삭제

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Selecting Series data

# ### How to do it...

# In[2]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
city = college['CITY']
city


# In[3]:


city['Alabama A & M University']


# In[4]:


city.loc['Alabama A & M University']


# In[5]:


city.iloc[0]


# In[6]:


city[['Alabama A & M University', 'Alabama State University']]
city['Alabama A & M University', 'Alabama State University'] # 에러남
type(city[['Alabama A & M University', 'Alabama State University']]) # pandas.core.series.Series


# In[7]:


city.loc[['Alabama A & M University', 'Alabama State University']]


# In[8]:


city.iloc[[0, 4]]


# In[9]:


city['Alabama A & M University': 'Alabama State University']


# In[10]:


city[0:5]


# In[11]:


city.loc['Alabama A & M University': 'Alabama State University']


# In[12]:


city.iloc[0:5]


# In[13]:

# Boolean arrays
alabama_mask = city.isin(['Birmingham', 'Montgomery'])
type(alabama_mask)
city[alabama_mask]


# ### How it works...

# In[14]:


s = pd.Series([10, 20, 35, 28], index=[5,2,3,1])
s


# In[15]:


s[0:4]


# In[16]:


s[5]


# In[17]:


s[1]


# ### There's more...

# In[18]:


college.loc['Alabama A & M University', 'CITY']
type(college)
type(college.loc['Alabama A & M University', 'CITY'])



# In[19]:


college.iloc[0, 0]


# In[20]:


college.loc[['Alabama A & M University', 
             'Alabama State University'], 'CITY']


# In[21]:


college.iloc[[0, 4], 0]


# In[22]:


college.loc['Alabama A & M University':
            'Alabama State University', 'CITY']


# In[23]:


college.iloc[0:5, 0]


# In[24]:


city.loc['Reid State Technical College':
         'Alabama State University']


# ## Selecting DataFrame rows

# In[25]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.sample(5, random_state=42)


# In[26]:


college.iloc[60]


# In[27]:


college.loc['University of Alaska Anchorage']


# In[28]:


college.iloc[[60, 99, 3]]


# In[29]:


labels = ['University of Alaska Anchorage',
          'International Academy of Hair Design',
          'University of Alabama in Huntsville']
college.loc[labels]


# In[30]:


college.iloc[99:102]


# In[31]:


start = 'International Academy of Hair Design'
stop = 'Mesa Community College'
college.loc[start:stop]


# ### How it works...

# ### There's more...

# In[32]:


college.iloc[[60, 99, 3]].index.tolist()


# ## Selecting DataFrame rows and columns simultaneously

# ### How to do it...

# In[33]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.iloc[:3, :4]


# In[34]:


college.loc[:'Amridge University', :'MENONLY']


# In[35]:


college.iloc[:, [4,6]].head()


# In[36]:


college.loc[:, ['WOMENONLY', 'SATVRMID']].head()


# In[37]:


college.iloc[[100, 200], [7, 15]]


# In[38]:


rows = ['GateWay Community College',
        'American Baptist Seminary of the West']
columns = ['SATMTMID', 'UGDS_NHPI']
college.loc[rows, columns]


# In[39]:


college.iloc[5, -4]


# In[40]:


college.loc['The University of Alabama', 'PCTFLOAN']


# In[41]:


college.iloc[90:80:-2, 5]


# In[42]:


start = 'Empire Beauty School-Flagstaff'
stop = 'Arizona State University-Tempe'
college.loc[start:stop:-2, 'RELAFFIL']


# ### How it works...

# ### There's more...

# ## Selecting data with both integers and labels

# ### How to do it...

# In[43]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')


# In[44]:


col_start = college.columns.get_loc('UGDS_WHITE')
col_end = college.columns.get_loc('UGDS_UNKN') + 1
col_start, col_end


# In[45]:


college.iloc[:5, col_start:col_end]


# ### How it works...

# ### There's more...

# In[46]:


row_start = college.index[10]
row_end = college.index[15]
college.loc[row_start:row_end, 'UGDS_WHITE':'UGDS_UNKN']


# In[47]:


college.ix[10:16, 'UGDS_WHITE':'UGDS_UNKN']


# In[48]:


college.iloc[10:16].loc[:, 'UGDS_WHITE':'UGDS_UNKN']


# ## Slicing lexicographically

# ### How to do it...

# In[49]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')


# In[50]:


college.loc['Sp':'Su']


# In[51]:


college = college.sort_index()


# In[52]:


college.loc['Sp':'Su']

college.sort_index().loc['Sp':'Su']
# ### How it works...

# ### There's more...

# In[53]:


college = college.sort_index(ascending=False)
college.index.is_monotonic_decreasing


# In[54]:

college.loc['E':'B']

college.loc['A':'B'] # 아무것도 안들어옴
 
# In[ ]:






'''  code07_7장_행 필터링.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Filtering Rows

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name) # name object 삭제

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Calculating boolean statistics

# ### How to do it...

# In[2]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie[['duration']].head()


# In[3]:


movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)


# In[4]:


movie_2_hours.sum()


# In[5]:


movie_2_hours.mean()


# In[6]:


movie['duration'].dropna().gt(120).mean()


# In[7]:


movie_2_hours.describe()


# ### How it works...

# In[8]:


movie_2_hours.value_counts(normalize=True)


# In[9]:


movie_2_hours.astype(int).describe()


# ### There's more...

# In[10]:


actors = movie[['actor_1_facebook_likes',
                'actor_2_facebook_likes']].dropna()

(actors['actor_1_facebook_likes'] >
      actors['actor_2_facebook_likes']).mean()


# ## Constructing multiple boolean conditions

# ### How to do it...

# In[11]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')


# In[12]:


criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = ((movie.title_year < 2000) | (movie.title_year > 2009))


# In[13]:


criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()


# ### How it works...
 
# ### There's more...

# In[14]:


5 < 10 and 3 > 4


# In[15]:


5 < 10 and 3 > 4


# In[16]:


True and 3 > 4


# In[17]:


True and False


# In[18]:


False


# In[19]:


movie.title_year < 2000 | movie.title_year > 2009


# ## Filtering with boolean arrays

# ### How to do it...

# In[20]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
crit_a1 = movie.imdb_score > 8
crit_a2 = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3


# In[21]:


crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = ((movie.title_year >= 2000) &
(movie.title_year <= 2010))
final_crit_b = crit_b1 & crit_b2 & crit_b3


# In[22]:


final_crit_all = final_crit_a | final_crit_b
final_crit_all.head()


# In[23]:


movie[final_crit_all].head()


# In[24]:


movie.loc[final_crit_all].head()
type(final_crit_all) # pandas.core.series.Series

# integer location method에서는 tolist() 또는 to_numpy()로 변환하여
# 리스트나 배열 형태의 불리언 어레이를 만들어 넣어줘야 함
movie.iloc[final_crit_all.tolist(), ].head() 
movie.iloc[final_crit_all.to_numpy(), ].head()
type(final_crit_all.tolist()) # list
type(final_crit_all.to_numpy()) # numpy.ndarray

# 또는 Boolean의 값만 가지고 와서 사용
movie.iloc[final_crit_all.values].head()
type(final_crit_all.values)
# In[25]:


cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols]
movie_filtered.head(10)


# ### How it works...

# In[26]:


movie.iloc[final_crit_all]


# In[43]:


movie.iloc[final_crit_all.values]


# ### There's more...

# In[44]:


final_crit_a2 = ((movie.imdb_score > 8) & 
   (movie.content_rating == 'PG-13') & 
   ((movie.title_year < 2000) |
    (movie.title_year > 2009))
)
final_crit_a2.equals(final_crit_a)


# ## Comparing Row Filtering and Index Filtering

# ### How to do it...

# In[45]:


college = pd.read_csv('data/college.csv')
college[college['STABBR'] == 'TX'].head()


# In[46]:


college2 = college.set_index('STABBR')
college2.loc['TX'].head()


# In[47]:


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


# In[48]:


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


# In[49]:


get_ipython().run_line_magic('timeit', "college2 = college.set_index('STABBR')")


# ### How it works...

# ### There's more...

# In[50]:


states = ['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)]


# In[51]:


college2.loc[states]


# ## Selecting with unique and sorted indexes

# ### How to do it...

# In[52]:


college = pd.read_csv('data/college.csv')
college2 = college.set_index('STABBR')
college2.index.is_monotonic


# In[53]:


college3 = college2.sort_index()
college3.index.is_monotonic


# In[54]:


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


# In[55]:


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


# In[56]:


get_ipython().run_line_magic('timeit', "college3.loc['TX']")


# In[57]:


college_unique = college.set_index('INSTNM')
college_unique.index.is_unique


# In[58]:


college[college['INSTNM'] == 'Stanford University']


# In[59]:


college_unique.loc['Stanford University']


# In[60]:


college_unique.loc[['Stanford University']]


# In[61]:


get_ipython().run_line_magic('timeit', "college[college['INSTNM'] == 'Stanford University']")


# In[62]:


get_ipython().run_line_magic('timeit', "college_unique.loc[['Stanford University']]")


# ### How it works...

# ### There's more...

# In[63]:
college.iloc[:, 0:2].iloc[:, [True, False]]

college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()


# In[64]:


college.loc['Miami, FL'].head()


# In[65]:


get_ipython().run_cell_magic('timeit', 
                             '', 
                             "crit1 = college['CITY'] == 'Miami'\ncrit2 = college['STABBR'] == 'FL'\ncollege[crit1 & crit2]")


# In[66]:


get_ipython().run_line_magic('timeit', 
                             "college.loc['Miami, FL']")


# ## Translating SQL WHERE clauses

# ### How to do it...

# In[67]:


employee = pd.read_csv('data/employee.csv')


# In[68]:


employee.dtypes


# In[69]:


employee.DEPARTMENT.value_counts().head()


# In[70]:


employee.GENDER.value_counts()


# In[71]:


employee.BASE_SALARY.describe()


# In[72]:


depts = ['Houston Police Department-HPD',
         'Houston Fire Department (HFD)']

criteria_dept = employee.DEPARTMENT.isin(depts)

criteria_gender = employee.GENDER == 'Female'

criteria_sal = ((employee.BASE_SALARY >= 80000) & (employee.BASE_SALARY <= 120000))


# In[73]:


criteria_final = (criteria_dept &
                  criteria_gender &
                  criteria_sal)
type(criteria_final) # pandas.core.series.Series

criteria_final_test = [criteria_dept &
                       criteria_gender &
                       criteria_sal]
type(criteria_final_test) # list


# In[74]:


select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']

employee.loc[criteria_final, select_columns].head()


# ### How it works...

# ### There's more...

# In[75]:


criteria_sal = employee.BASE_SALARY.between(80_000, 120_000)


# In[76]:


top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria]


# ## Improving readability of boolean indexing with the query method

# ### How to do it...

# In[77]:


employee = pd.read_csv('data/employee.csv')
depts = ['Houston Police Department-HPD',
         'Houston Fire Department (HFD)']
select_columns = ['UNIQUE_ID', 
                  'DEPARTMENT',
                  'GENDER', 
                  'BASE_SALARY']


# In[78]:


qs =( "DEPARTMENT in @depts "
      " and GENDER == 'Female' "
      " and 80000 <= BASE_SALARY <= 120000" )
emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()


# ### How it works...

# ### There's more...

# In[79]:


top10_depts = (employee.DEPARTMENT.value_counts() 
   .index[:10].tolist()
)
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2.head()


# ## Preserving Series size with the where method

# ### How to do it...

# In[80]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()
type(fb_likes)

# In[81]:


fb_likes.describe()


# In[82]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fb_likes.hist(ax=ax)
fig.savefig('tmp/c7-hist.png', dpi=300)     # doctest: +SKIP


# In[83]:


criteria_high = fb_likes < 20_000
criteria_high.mean().round(2)


# In[84]:


fb_likes.where(criteria_high).head()


# In[85]:


fb_likes.where(criteria_high, other = 20000).head()



# In[86]:


criteria_low = fb_likes > 300
fb_likes_cap = (fb_likes
   .where(criteria_high, other=20_000)
   .where(criteria_low, 300)
)
fb_likes_cap.head()


# In[87]:


len(fb_likes), len(fb_likes_cap)


# In[88]:


fig, ax = plt.subplots(figsize=(10, 8))
fb_likes_cap.hist(ax=ax)
fig.savefig('tmp/c7-hist2.png', dpi=300)     # doctest: +SKIP


# ### How it works...

# ### There's more...

# In[89]:


fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)


# ## Masking DataFrame rows

# ### How to do it...

# In[90]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isna()
criteria = c1 | c2

movie.title_year

# In[91]:


movie.title_year.mask(criteria).head()


# In[92]:


movie_mask = (movie
    .mask(criteria)
    .dropna(how='all')
)
movie_mask.head()


# In[93]:


movie_boolean = movie[movie['title_year'] < 2010]
movie_mask.equals(movie_boolean)


# In[94]:


movie_mask.shape == movie_boolean.shape


# In[95]:


movie_mask.dtypes == movie_boolean.dtypes
type(movie_mask.dtypes != movie_boolean.dtypes)


movie_mask.loc[:, movie_mask.dtypes != movie_boolean.dtypes]

movie_mask.iloc[:, (movie_mask.dtypes != movie_boolean.dtypes).tolist()]

movie_mask.dtypes[movie_mask.dtypes != movie_boolean.dtypes]
movie_boolean.dtypes[movie_mask.dtypes != movie_boolean.dtypes]


# In[96]:


from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask, check_dtype=False)
# 같으면 None 반환

assert_frame_equal(movie_boolean, movie_mask, check_dtype=True)
# 같지 않으므로 오류 발생 
# ### How it works...

# ### There's more...

# In[97]:


get_ipython().run_line_magic('timeit', "movie.mask(criteria).dropna(how='all')")


# In[98]:


get_ipython().run_line_magic('timeit', "movie[movie['title_year'] < 2010]")


# ## Selecting with booleans, integer location, and labels

# ### How to do it...

# In[99]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2
type(criteria)

# In[100]:


movie_loc = movie.loc[criteria]
movie_loc.head()


# In[101]:


movie_loc.equals(movie[criteria])


# In[102]:


movie_iloc = movie.iloc[criteria]


# In[103]:


movie_iloc = movie.iloc[criteria.values]
movie_iloc.equals(movie_loc)


# In[104]:


criteria_col = movie.dtypes == np.int64

criteria_col.head()
type(criteria_col) # Series


# In[105]:


movie.loc[:, criteria_col].head()


# In[106]:


movie.iloc[:, criteria_col.values].head()


# In[107]:


cols = ['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')


# In[108]:


col_index = [movie.columns.get_loc(col) for col in cols]
col_index


# In[109]:


movie.iloc[criteria.values, col_index].sort_values('imdb_score')


# ### How it works...

# In[110]:


a = criteria.values
a[:5]


# In[111]:


len(a), len(criteria)
type(a), type(criteria)

# In[112]:

movie.dtypes.value_counts()

movie.select_dtypes(include = "int64")
movie.select_dtypes(object)
movie.select_dtypes(float)
movie.dtypes
type(movie)

# In[ ]:

movie.columns.isin(movie.columns.sort_values()[0:2])
movi.columnslike




'''  code08_8장_인덱스 정렬.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Index Alignment
import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        
del(name)

# In[3]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 50)


# ## Introduction

# ## Examining the Index object

# ### How to do it...

# In[12]:


college = pd.read_csv('data/college.csv')
columns = college.columns
columns


# In[13]:


columns.values

type(columns) # pandas.core.indexes.base.Index

# In[14]:


columns[5]


# In[15]:


columns[[1,8,10]]


# In[16]:


columns[-7:-4]


# In[17]:


columns.min(), columns.max(), columns.isnull().sum()


# In[18]:


columns + '_A'


# In[19]:


columns > 'G'


# In[20]:


columns[1] = 'city'


# ### How it works...

# ### There's more...

# In[61]:


c1 = columns[:4]
c1


# In[62]:


c2 = columns[2:6]
c2


# In[63]:


c1.union(c2) # or `c1 | c2`
c1|c2

# In[64]:

# 합집합 - 교집합 
c1.symmetric_difference(c2) # or `c1 ^ c2`


# ## Producing Cartesian products

# ### How to do it...

# In[65]:


s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1


# In[66]:


s2 = pd.Series(index=list('cababb'), data=np.arange(6))
s2


# In[67]:


s1 + s2


# ### How it works...

# ### There's more...

# In[68]:


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1 + s2


# In[69]:


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1 + s2


# In[70]:


s3 = pd.Series(index=list('ab'), data=np.arange(2))
s4 = pd.Series(index=list('ba'), data=np.arange(2))
s3 + s4


# ## Exploding indexes

# ### How to do it...

# In[71]:


employee = pd.read_csv('data/employee.csv', index_col='RACE')
employee.head()


# In[72]:


salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2


# In[73]:


salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2

employee['BASE_SALARY'] is salary1


# In[74]:


salary1 = salary1.sort_index()
salary1.head()


# In[75]:


salary2.head()


# In[76]:


salary_add = salary1 + salary2


# In[77]:


salary_add.head()


# In[78]:


salary_add1 = salary1 + salary1
len(salary1), len(salary2), len(salary_add), len(salary_add1)


# ### How it works...

# ### There's more...

# In[79]:


index_vc = salary1.index.value_counts(dropna=False)
(index_vc**2).sum()


# In[80]:


index_vc.pow(2).sum()


# ## Filling values with unequal indexes

# In[4]:


baseball_14 = pd.read_csv('data/baseball14.csv',
   index_col='playerID')
baseball_15 = pd.read_csv('data/baseball15.csv',
   index_col='playerID')
baseball_16 = pd.read_csv('data/baseball16.csv',
   index_col='playerID')
baseball_14.head()


# In[82]:


baseball_14.index.difference(baseball_15.index)


# In[83]:


baseball_14.index.difference(baseball_16.index)


# In[84]:


hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()


# In[85]:


(hits_14 + hits_15).head()


# In[86]:

# 결측치는 0으로 채우고 합산
hits_14.add(hits_15, fill_value=0).head()


# In[87]:


hits_total = (hits_14
   .add(hits_15, fill_value=0)
   .add(hits_16, fill_value=0)
)
hits_total.head()


# In[88]:


hits_total.hasnans


# ### How it works...

# In[89]:


s = pd.Series(index=['a', 'b', 'c', 'd'],
              data=[np.nan, 3, np.nan, 1])
s


# In[90]:


s1 = pd.Series(index=['a', 'b', 'c'], data=[np.nan, 6, 10])
s1


# In[91]:


s.add(s1, fill_value=5)


# ### There's more...

# In[5]:


df_14 = baseball_14[['G','AB', 'R', 'H']]
df_14.head()


# In[6]:


df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()


# In[7]:


(df_14 + df_15).head(10).style.highlight_null('yellow')


# In[8]:


(df_14
.add(df_15, fill_value=0)
.head(10)
.style.highlight_null('yellow')
)


# ## Adding columns from different DataFrames

# ### How to do it...

# In[94]:


employee = pd.read_csv('data/employee.csv')
dept_sal = employee[['DEPARTMENT', 'BASE_SALARY']]


# In[95]:


dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'],
    ascending=[True, False])


# In[96]:

# 부서별 최대값만 남기고 제거함 
max_dept_sal = dept_sal.drop_duplicates(subset='DEPARTMENT')
max_dept_sal.head()


# In[97]:

# 부서 행을 인덱스로 일치시켜 줌
max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
employee = employee.set_index('DEPARTMENT')


# In[98]:

employee = (employee
   .assign(MAX_DEPT_SALARY = max_dept_sal['BASE_SALARY'])
)

employee
employee.shape
len(employee.index)


# In[99]:


employee.query('BASE_SALARY > MAX_DEPT_SALARY')


# In[100]:


employee = pd.read_csv('data/employee.csv')
max_dept_sal = (employee
    [['DEPARTMENT', 'BASE_SALARY']]
    .sort_values(['DEPARTMENT', 'BASE_SALARY'],
        ascending=[True, False])
    .drop_duplicates(subset = 'DEPARTMENT')
    .set_index('DEPARTMENT')
)


# In[101]:


(employee
   .set_index('DEPARTMENT')
   .assign(MAX_DEPT_SALARY=max_dept_sal['BASE_SALARY'])
)


# ### How it works...

# In[102]:


random_salary = (dept_sal
    .sample(n=10, random_state=42)
    .set_index('DEPARTMENT')
)
random_salary


# In[103]:


employee['RANDOM_SALARY'] = random_salary['BASE_SALARY']


# ### There's more...

# In[104]:


(employee
    .set_index('DEPARTMENT')
    .assign(MAX_SALARY2=max_dept_sal['BASE_SALARY'].head(3))
    .MAX_SALARY2.index
    # .value_counts()
)

max_dept_sal

# In[105]:


max_sal = (employee
    .groupby('DEPARTMENT')
    .BASE_SALARY
    .transform('max') # 원래 index를 유지하는 메서드
)

max_sal.index

# In[106]:


(employee
    .assign(MAX_DEPT_SALARY=max_sal)
)


# In[107]:


max_sal = (employee
    .groupby('DEPARTMENT')
    .BASE_SALARY
    .max()
)

max_sal.index
employee.columns

# In[108]:


(employee
    .merge(max_sal.rename('MAX_DEPT_SALARY'),
           how='left', left_on='DEPARTMENT',
           right_index=True)
)


# ## Highlighting the maximum value from each column

# ### How to do it...

# In[9]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.dtypes


# In[110]:


college.MD_EARN_WNE_P10.sample(10, random_state=42)


# In[111]:


college.GRAD_DEBT_MDN_SUPP.sample(10, random_state=42)


# In[112]:


college.MD_EARN_WNE_P10.value_counts()


# In[113]:


set(college.MD_EARN_WNE_P10.apply(type))


# In[114]:


college.GRAD_DEBT_MDN_SUPP.value_counts()


# In[115]:

# 수치형으로 변경
cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')


# In[116]:


college.dtypes.loc[cols]


# In[11]:


college_n = college.select_dtypes('number')
college_n.head()


# In[13]:


binary_only = college_n.nunique() == 2
binary_only.head()


# In[14]:

# Binary only에서 True 컬럼만 추출 
binary_cols = binary_only[binary_only].index.tolist()
binary_cols


# In[15]:

# 이진열 삭제
college_n2 = college_n.drop(columns=binary_cols)
college_n2.head()


# In[16]:

# .idxmax 메소드: 각 열의 최대값의 인덱스를 추출
max_cols = college_n2.idxmax()
max_cols


# In[17]:


unique_max_cols = max_cols.unique()
unique_max_cols[:5]


# In[123]:


college_n2.loc[unique_max_cols] #.style.highlight_max()


# In[18]:


college_n2.loc[unique_max_cols].style.highlight_max()

# In[124]:

# 코드 정리
def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)


# In[125]:


def select_rows_with_max_cols(df):
    max_cols = df.idxmax()
    unique = max_cols.unique()
    return df.loc[unique]


# In[126]:


(college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
   .pipe(select_rows_with_max_cols)
)


# ### How it works...

# ### There's more...

# In[19]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()


# In[20]:


college_ugds.style.highlight_max(axis='columns')


# ## Replicating idxmax with method chaining

# ### How to do it...

# In[128]:


def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)


# In[129]:


college_n = (college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
)


# In[130]:


college_n.max().head()


# In[131]:


college_n.eq(college_n.max()).head()


# In[132]:


has_row_max = (college_n
    .eq(college_n.max())
    .any(axis='columns')
)
has_row_max.head()


# In[133]:


college_n.shape


# In[134]:


has_row_max.sum()


# In[135]:


college_n.eq(college_n.max()).cumsum()


# In[136]:


(college_n
    .eq(college_n.max())
    .cumsum()
    .cumsum()
)


# In[137]:


has_row_max2 = (college_n
    .eq(college_n.max()) 
    .cumsum() 
    .cumsum() 
    .eq(1) 
    .any(axis='columns')
)


# In[138]:


has_row_max2.head()


# In[139]:


has_row_max2.sum()


# In[140]:


idxmax_cols = has_row_max2[has_row_max2].index
idxmax_cols


# In[141]:


set(college_n.idxmax().unique()) == set(idxmax_cols)


# In[142]:


def idx_max(df):
     has_row_max = (df
         .eq(df.max())
         .cumsum()
         .cumsum()
         .eq(1)
         .any(axis='columns')
     )
     return has_row_max[has_row_max].index


# In[143]:


idx_max(college_n)
len(idx_max(college_n))

len(college_n.idxmax().values)
# ### How it works...

# ### There's more...

# In[144]:


def idx_max(df):
     has_row_max = (df
         .eq(df.max())
         .cumsum()
         .cumsum()
         .eq(1)
         .any(axis='columns')
         [lambda df_: df_]
         .index
     )
     return has_row_max


# In[145]:


get_ipython().run_line_magic('timeit', 'college_n.idxmax().values')


# In[146]:


get_ipython().run_line_magic('timeit', 'idx_max(college_n)')


# ## Finding the most common maximum of columns

# ### How to do it...

# In[147]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


# In[148]:


highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.head()


# In[149]:


highest_percentage_race.value_counts(normalize=True)


# ### How it works...

# ### There's more...

# In[150]:

 

# In[ ]:






'''  code09_9장_그룹화를 위한 집계, 여과, 변환.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Grouping for Aggregation, Filtration, and Transformation

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del(name)


# In[ ]:

import pandas as pd
import numpy as np
pd.set_option('max_columns', 10, 'max_rows', 10, 'max_colwidth', 50)


# ## Introduction

# ### Defining an Aggregation

# ### How to do it...

# In[ ]:


flights = pd.read_csv('data/flights.csv')
flights.head()


# In[ ]:


(flights
     .groupby('AIRLINE')
     .agg({'ARR_DELAY':'mean'})
)

test = flights.groupby('AIRLINE').agg({'ARR_DELAY':'mean'})

type(test)

# In[ ]:


(flights
     .groupby('AIRLINE')
     ['ARR_DELAY']
      .agg('mean')
)


# In[ ]:


(flights
    .groupby('AIRLINE')
    ['ARR_DELAY']
    .agg(np.mean)
)


# In[ ]:


(flights
    .groupby('AIRLINE')
    ['ARR_DELAY']
    .mean()
)


# ### How it works...

# In[ ]:


grouped = flights.groupby('AIRLINE')
type(grouped)


# ### There's more...

# In[ ]:


(flights
   .groupby('AIRLINE')
   ['ARR_DELAY']
   .agg(np.sqrt)
)


# ## Grouping and aggregating with multiple columns and functions

# ### How to do it...

# In[ ]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    ['CANCELLED'] 
    .agg('sum')
)


# In[ ]:

# 하지만 책에서는 집계열의 쌍에 대해서도 리스트 사용을 권장
(flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    [['CANCELLED', 'DIVERTED']]
    .agg(['sum', 'mean'])
)


(flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    [['CANCELLED', 'DIVERTED']]
    .agg(['sum', 'mean'])
).equals((flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    ['CANCELLED', 'DIVERTED']
    .agg(['sum', 'mean'])
))


# In[ ]:


(flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['count', 'sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)

flights.CANCELLED.value_counts()
flights.DIVERTED.value_counts()

flights.columns





# In[ ]:


(flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg(sum_cancelled = pd.NamedAgg(column = 'CANCELLED', aggfunc = 'sum'),
         mean_cancelled = pd.NamedAgg(column = 'CANCELLED', aggfunc = 'mean'),
         size_cancelled = pd.NamedAgg(column = 'CANCELLED', aggfunc = 'size'),
         mean_air_time = pd.NamedAgg(column = 'AIR_TIME', aggfunc = 'mean'),
         var_air_time = pd.NamedAgg(column = 'AIR_TIME', aggfunc = 'var'))
)


# ### How it works...

# ### There's more...

# In[ ]:


res = (flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)

res.columns = ['_'.join(x) for x in
    res.columns.to_flat_index()]

type(res.columns.to_flat_index())

# In[ ]:


res


# In[ ]:


def flatten_cols(df):
    df.columns = ['_'.join(x) for x in
        df.columns.to_flat_index()]
    return df


# In[ ]:


res = (flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
    .pipe(flatten_cols)# .reindex 메서드는 펼치기를 지원하지 않으므로 
                       # .pipe 메서드를 활용(위에 정의한 flatten_cols 함수 이용)
)


# In[ ]:


res


# In[ ]:


res = (flights
    .assign(ORG_AIR=flights.ORG_AIR.astype('category')) 
     # 그룹화 열 중 하나가 범주형(category)이면 카티션곱(모든 조합) 발생
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)
res

flights.ORG_AIR.value_counts()
flights.DEST_AIR.value_counts()

test = flights.assign(ORG_AIR=flights.ORG_AIR.astype('category'))
test.ORG_AIR.dtypes




# In[ ]:


res = (flights
    .assign(ORG_AIR=flights.ORG_AIR.astype('category'))
    .groupby(['ORG_AIR', 'DEST_AIR'], observed=True)
     # 모든 조합(카티션곱 폭발)을 방지하려면 observed = True 인자 사용 
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)
res


# ## Removing the MultiIndex after grouping

# In[ ]:

    # 그룹화 후 다중 인덱스 제거

flights = pd.read_csv('data/flights.csv')
airline_info = (flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    .agg({'DIST':['sum', 'mean'],
          'ARR_DELAY':['min', 'max']}) 
    .astype(int)
)
 
airline_info


# In[ ]:


airline_info.columns.get_level_values(0)
airline_info.columns.get_level_values(0).dtype

# In[ ]:


airline_info.columns.get_level_values(1)
airline_info.index.get_level_values(1)
airline_info.index.get_level_values(1)

# In[ ]:


airline_info.columns.to_flat_index()
airline_info.index.to_flat_index()

# In[ ]:


airline_info.columns = ['_'.join(x) for x in
    airline_info.columns.to_flat_index()]

# airline_info.index = ['_'.join(x) for x in
#     airline_info.index.to_flat_index()]


# In[ ]:


airline_info


# In[ ]:


airline_info.reset_index()


# In[ ]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    .agg(dist_sum=pd.NamedAgg(column='DIST', aggfunc='sum'),
         dist_mean=pd.NamedAgg(column='DIST', aggfunc='mean'),
         arr_delay_min=pd.NamedAgg(column='ARR_DELAY', aggfunc='min'),
         arr_delay_max=pd.NamedAgg(column='ARR_DELAY', aggfunc='max'))
    .astype(int)
    .reset_index()
)

flights.AIRLINE.dtype
flights.WEEKDAY.dtype


# ### How it works...

# ### There's more...

# In[ ]:


(flights
    .groupby(['AIRLINE'] , as_index=False)
    ['DIST']
    .agg('mean')
    .round(0)
)


(flights
    .groupby(['AIRLINE'])
    ['DIST']
    .agg('mean')
    .round(0)
)

(flights
    .groupby(['AIRLINE'], sort = False)
    ['DIST']
    .agg('mean')
    .round(0)
)

(
 get_ipython()
 .run_line_magic('timeit',
                 'flights.groupby(["AIRLINE"])["DIST"].agg("mean").round(0)')
 )

# groupby 안에 sort 기능을 사용안하면 약간의 성능이 향상됨 
(get_ipython()
.run_line_magic('timeit', 
'(flights.groupby(["AIRLINE"], sort = False)["DIST"].agg("mean").round(0))')
)

# ## Grouping with a custom aggregation function

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv')

(college
    .groupby('STABBR')
    ['UGDS']
    .agg(['mean', 'std'])
    .round(0)
)


# In[ ]:


def max_deviation(s):
    std_score = (s - s.mean()) / s.std()
    return std_score.abs().max()


# In[ ]:


(college
    .groupby('STABBR')
    ['UGDS']
    .agg(max_deviation)
    .round(1)
)

tmp = (college
    .groupby('STABBR')
    ['UGDS']
    .agg(max_deviation)
    .round(1)
)

tmp.max()
tmp.idxmax()
type(tmp[tmp.index == 'AS'])

tmp[tmp.index == tmp.idxmax()]
type(tmp[tmp.index == tmp.idxmax()])


# ### How it works...

# ### There's more...

# In[ ]:


(college
    .groupby('STABBR')
    ['UGDS', 'SATVRMID', 'SATMTMID']
    .agg(max_deviation)
    .round(1)
)


# In[ ]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATVRMID', 'SATMTMID'] 
    .agg([max_deviation, 'mean', 'std'])
    .round(1)
)


# In[ ]:


max_deviation.__name__


# In[ ]:


max_deviation.__name__ = 'Max Deviation'
tmp2 = (college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATVRMID', 'SATMTMID'] 
    .agg([max_deviation, 'mean', 'std'])
    .round(1)
)

tmp2.columns
tmp2.columns.rename({('UGDS', 'Max Deviation'),('UGDS', 'MD')}, inplace = True)



# ## Customizing aggregating functions with *args and **kwargs

# ### How to do it...

# In[ ]:

# 학부생 비율이 1,000에서 3,000 사이인 학교의 비율을 반환하는 함수 정의

def pct_between_1_3k(s):
    return (s
        .between(1_000, 3_000)
        .mean()
        * 100
    )


# In[ ]:
    
# 주와 종교에 대해 그룹화하고 비율을 계산함

(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between_1_3k)
    .round(1)
)


# In[ ]: 
    
# 상한과 하한을 사용자가 지정할 수 있는 함수 작성
# 상한과 하한을 지정하여 그 비율을 산출하는 함수 작성

def pct_between(s, low, high):
    return s.between(low, high).mean() * 100


# In[ ]:

# 1,000 ~ 10,000으로 범위 지정

(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between, 1_000, 10_000)
    .round(1)
)

# 명시적으로 키워드 매개변수를 사용해 동일한 결과를 얻을 수 있다.
(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between, high = 10_000, low = 1_000)
    .round(1)
)


# ### How it works...

# ### There's more...

# In[ ]:

# 복수 집계함수를 호출하면서 일부 매개변수를 직접 제공하고 싶다면,
# 파이썬의 클로져 기능을 사용해 매개변수가 호출 환경에서 닫힌 상태로 되는
# 새로운 함수를 생성

def between_n_m(n, m):
    def wrapper(ser):
        return pct_between(ser, n, m)
    wrapper.__name__ = f'between_{n}_{m}'
    return wrapper


# In[ ]:


(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg([between_n_m(1_000, 10_000), 'max', 'mean'])
    .round(1)
)


# ## Examining the groupby object

# ### How to do it...

# In[ ]: 


college = pd.read_csv('data/college.csv')
grouped = college.groupby(['STABBR', 'RELAFFIL'])
type(grouped)

college.RELAFFIL.value_counts()

# In[ ]:


print([attr for attr in dir(grouped) if not
    attr.startswith('_')])


# In[ ]:


grouped.ngroups


# In[ ]:


groups = list(grouped.groups)
groups[:6]


# In[ ]:

grouped.get_group(('FL', 1))




# In[ ]:


from IPython.display import display
for name, group in grouped:
    print(name)
    display(group.head(3))


# In[ ]:

# 그룹별로 잘려진 데이터들을 확인하는 방법

for name, group in grouped:
    print(name)
    print(group)
    break


# In[ ]:


grouped.head(2)


# ### How it works...

# ### There's more...

# In[ ]: 정수리스트가 제공될 때, 각 그룹에서 해당 행을 선택하는 .nth 메서드 사용

grouped.nth([1, -1]) # 각 그룹에서 첫 번째와 마지막 행을 선택함


# ## Filtering for states with a minority majority

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
grouped = college.groupby('STABBR')
grouped.ngroups


# In[ ]:


college['STABBR'].nunique() # verifying the same number


# In[ ]:


def check_minority(df, threshold):
    minority_pct = 1 - df['UGDS_WHITE']
    total_minority = (df['UGDS'] * minority_pct).sum()
    total_ugds = df['UGDS'].sum()
    total_minority_pct = total_minority / total_ugds
    return total_minority_pct > threshold


# In[ ]:


college_filtered = grouped.filter(check_minority, threshold=.5)
college_filtered


# In[ ]:


college.shape


# In[ ]:


college_filtered.shape


# In[ ]:


college_filtered['STABBR'].nunique()


# ### How it works...

# ### There's more...

# In[ ]:


college_filtered_20 = grouped.filter(check_minority, threshold=.2)
college_filtered_20.shape


# In[ ]:


college_filtered_20['STABBR'].nunique()


# In[ ]:


college_filtered_70 = grouped.filter(check_minority, threshold=.7)
college_filtered_70.shape


# In[ ]:


college_filtered_70['STABBR'].nunique()


# ## Transforming through a weight loss bet

# ### How to do it...

# In[ ]:


weight_loss = pd.read_csv('data/weight_loss.csv')
weight_loss.query('Month == "Jan"')


# In[ ]:


def percent_loss(s):
    return ((s - s.iloc[0]) / s.iloc[0]) * 100


# In[ ]:


(weight_loss
    .query('Name=="Bob" and Month=="Jan"')
    ['Weight']
    .pipe(percent_loss)
)


# In[ ]:


(weight_loss
    .groupby(['Name', 'Month'])
    ['Weight'] 
    .transform(percent_loss)
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Name=="Bob" and Month in ["Jan", "Feb"]')
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
    .style.highlight_min(axis=1)
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
    .winner
    .value_counts()
)


# ### How it works...

# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .groupby(['Month', 'Name'])
    ['percent_loss']
    .first()
    .unstack()
)


# ### There's more...

# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)),
            Month=pd.Categorical(weight_loss.Month,
                  categories=['Jan', 'Feb', 'Mar', 'Apr'],
                  ordered=True))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
)


# ## Calculating weighted mean SAT scores per state with apply

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv')
subset = ['UGDS', 'SATMTMID', 'SATVRMID']
college2 = college.dropna(subset=subset)
college.shape


# In[ ]:


college2.shape


# In[ ]:


def weighted_math_average(df):
    weighted_math = df['UGDS'] * df['SATMTMID']
    return int(weighted_math.sum() / df['UGDS'].sum())


# In[ ]:

college2.UGDS
college2.SATMTMID

college2.groupby('STABBR').apply(weighted_math_average)


# In[ ]:


(college2
    .groupby('STABBR')
    .agg(weighted_math_average)
)


# In[ ]:


(college2
    .groupby('STABBR')
    ['SATMTMID'] 
    .agg(weighted_math_average)
)


# In[ ]:


def weighted_average(df):
   weight_m = df['UGDS'] * df['SATMTMID']
   weight_v = df['UGDS'] * df['SATVRMID']
   wm_avg = weight_m.sum() / df['UGDS'].sum()
   wv_avg = weight_v.sum() / df['UGDS'].sum()
   data = {'w_math_avg': wm_avg,
           'w_verbal_avg': wv_avg,
           'math_avg': df['SATMTMID'].mean(),
           'verbal_avg': df['SATVRMID'].mean(),
           'count': len(df)
   }
   return pd.Series(data)

(college2
    .groupby('STABBR')
    .apply(weighted_average)
    .astype(int)
)


# ### How it works...

# In[ ]:


(college
    .groupby('STABBR')
    .apply(weighted_average)
)


# ### There's more...

# In[ ]:


from scipy.stats import gmean, hmean

def calculate_means(df):
    df_means = pd.DataFrame(index=['Arithmetic', 'Weighted',
                                   'Geometric', 'Harmonic'])
    cols = ['SATMTMID', 'SATVRMID']
    for col in cols:
        arithmetic = df[col].mean()
        weighted = np.average(df[col], weights=df['UGDS'])
        geometric = gmean(df[col])
        harmonic = hmean(df[col])
        df_means[col] = [arithmetic, weighted,
                         geometric, harmonic]
    df_means['count'] = len(df)
    return df_means.astype(int)


(college2
    .groupby('STABBR')
    .apply(calculate_means)
)


# ## Grouping by continuous variables

# ### How to do it...

# In[ ]:


flights = pd.read_csv('data/flights.csv')
flights


# In[ ]:


bins = [-np.inf, 200, 500, 1000, 2000, np.inf]
cuts = pd.cut(flights['DIST'], bins=bins)
cuts


# In[ ]:


cuts.value_counts()
cuts.value_counts(normalize = True).round(2)*100

# In[ ]:


(flights
    .groupby(cuts)
    ['AIRLINE']
    .value_counts(normalize=True) 
    .round(3)
)


# ### How it works...

# ### There's more...

# In[ ]:


(flights
  .groupby(cuts)
  ['AIR_TIME']
  .quantile(q=[.25, .5, .75]) 
  .div(60)
  .round(2)
)


# In[ ]:


labels=['Under an Hour', '1 Hour', '1-2 Hours',
        '2-4 Hours', '4+ Hours']

cuts2 = pd.cut(flights['DIST'], bins=bins, labels=labels)

(flights
   .groupby(cuts2)
   ['AIRLINE']
   .value_counts(normalize=True) 
   .round(3) 
   .unstack() 
)


# ## Counting the total number of flights between cities

# ### How to do it...

# In[ ]:


flights = pd.read_csv('data/flights.csv')
flights_ct = flights.groupby(['ORG_AIR', 'DEST_AIR']).size()
flights_ct


# In[ ]:


flights_ct.loc[[('ATL', 'IAH'), ('IAH', 'ATL')]]


# In[ ]:


f_part3 = (flights  # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']] 
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
)
f_part3


# In[ ]:


rename_dict = {0:'AIR1', 1:'AIR2'}  
(flights     # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
)


# In[ ]:


(flights     # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
  .loc[('ATL', 'IAH')]
)


# In[ ]:


(flights     # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
  .loc[('IAH', 'ATL')]
)


# ### How it works...

# ### There's more ...

# In[ ]:


data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])
data_sorted[:10]


# In[ ]:


flights_sort2 = pd.DataFrame(data_sorted, columns=['AIR1', 'AIR2'])
flights_sort2.equals(f_part3.rename(columns={0:'AIR1',
    1:'AIR2'}))


# %%timeit
# flights_sort = (flights   # doctest: +SKIP
#     [['ORG_AIR', 'DEST_AIR']] 
#    .apply(lambda ser:
#          ser.sort_values().reset_index(drop=True),
#          axis='columns')
# )

# In[ ]:


get_ipython().run_cell_magic('timeit', 
                             '', 
                             "data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])\nflights_sort2 = pd.DataFrame(data_sorted,\n    columns=['AIR1', 'AIR2'])")


# ## Finding the longest streak of on-time flights

# ### How to do it...

# In[ ]:


s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
s


# In[ ]:

s1 = s.cumsum()
s1


# In[ ]:


s.mul(s1)


# In[ ]:


s.mul(s1).diff()


# In[ ]:


(s
    .mul(s.cumsum())
    .diff()
    .where(lambda x: x < 0)
)


# In[ ]:


(s
    .mul(s.cumsum())
    .diff()
    .where(lambda x: x < 0)
    .ffill()
)


# In[ ]:


(s
    .mul(s.cumsum())
    .diff()
    .where(lambda x: x < 0)
    .ffill()
    .add(s.cumsum(), fill_value=0)
)


# In[ ]:


flights = pd.read_csv('data/flights.csv')
(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    [['AIRLINE', 'ORG_AIR', 'ON_TIME']]
)


# In[ ]:


def max_streak(s):
    s1 = s.cumsum()
    return (s
       .mul(s1)
       .diff()
       .where(lambda x: x < 0) 
       .ffill()
       .add(s1, fill_value=0)
       .max()
    )


# In[ ]:


(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    .sort_values(['MONTH', 'DAY', 'SCHED_DEP']) 
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['ON_TIME'] 
    .agg(['mean', 'size', max_streak])
    .round(2)
)


# ### How it works...

# ### There's more...

# In[ ]:


def max_delay_streak(df):
    df = df.reset_index(drop=True)
    late = 1 - df['ON_TIME']
    late_sum = late.cumsum()
    streak = (late
        .mul(late_sum)
        .diff()
        .where(lambda x: x < 0) 
        .ffill()
        .add(late_sum, fill_value=0)
    )
    last_idx = streak.idxmax()
    first_idx = last_idx - streak.max() + 1
    res = (df
        .loc[[first_idx, last_idx], ['MONTH', 'DAY']]
        .assign(streak=streak.max())
    )
    res.index = ['first', 'last']
    return res


# In[ ]:


(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    .sort_values(['MONTH', 'DAY', 'SCHED_DEP']) 
    .groupby(['AIRLINE', 'ORG_AIR']) 
    .apply(max_delay_streak) 
    .sort_values('streak', ascending=False)
)


# In[ ]:






'''  code10_10장_정돈된 형식으로 재구성(groupby).py  '''

#!/usr/bin/env python
# coding: utf-8

# # Restructuring Data into a Tidy Form

import os
os.getcwd()
os.chdir('D:/pandas_cookbook')




# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Tidying variable values as column names with stack

# In[2]:


state_fruit = pd.read_csv('data/state_fruit.csv', index_col=0)
state_fruit


# ### How to do it...

# In[3]:


state_fruit.stack()

stf = state_fruit.stack().copy()
stf.shape
type(stf)

# In[4]:

# reset_index를 이용해 Data Frame으로 변환
(state_fruit
   .stack()
   .reset_index()
)


# In[5]:


(state_fruit
   .stack()
   .reset_index()
   .rename(columns={'level_0':'state', 
      'level_1': 'fruit', 0: 'weight'})
)


# In[6]:


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
)


# In[7]:


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
    .reset_index(name='weight')
)

# ?pd.reset_index
# help(pd.Series.reset_index)

# ### How it works...

# ### There's more...

# In[8]:


state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


# In[9]:


state_fruit2.stack()


# In[10]:


state_fruit2.set_index('State').stack()


# ## Tidying variable values as column names with melt

# ### How to do it...

# In[11]:


state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


# In[12]:


state_fruit2.melt(id_vars=['State'],
    value_vars=['Apple', 'Orange', 'Banana'])


# In[13]:


state_fruit2.melt(id_vars=['State'],
                   value_vars=['Apple', 'Orange', 'Banana'],
                   var_name='Fruit',
                   value_name='Weight')


# ### How it works...

# ### There's more...

# In[14]:


state_fruit2.melt()


# In[15]:


state_fruit2.melt(id_vars='State')


# ## Stacking multiple groups of variables simultaneously

# In[16]:


movie = pd.read_csv('data/movie.csv')
actor = movie[['movie_title', 'actor_1_name',
               'actor_2_name', 'actor_3_name',
               'actor_1_facebook_likes',
               'actor_2_facebook_likes',
               'actor_3_facebook_likes']]
actor.head()


# ### How to do it...

# In[17]:


def change_col_name(col_name):
    col_name = col_name.replace('_name', '')
    if 'facebook' in col_name:
        fb_idx = col_name.find('facebook')
        col_name = (col_name[:5] + col_name[fb_idx - 1:] 
               + col_name[5:fb_idx-1])
    return col_name


# In[18]:


actor2 = actor.rename(columns=change_col_name)
actor2


# In[19]:


stubs = ['actor', 'actor_facebook_likes']
actor2_tidy = pd.wide_to_long(actor2,
    stubnames=stubs,
    i=['movie_title'],
    j='actor_num',
    sep='_')
actor2_tidy.head()


# ### How it works...

# ### There's more...

# In[20]:


df = pd.read_csv('data/stackme.csv')
df


# In[21]:


df.rename(columns = {'a1':'group1_a1', 'b2':'group1_b2',
                     'd':'group2_a1', 'e':'group2_b2'})


# In[22]:


pd.wide_to_long(
       df.rename(columns = {'a1':'group1_a1', 
                 'b2':'group1_b2',
                 'd':'group2_a1', 'e':'group2_b2'}),
    stubnames=['group1', 'group2'],
    i=['State', 'Country', 'Test'],
    j='Label',
    suffix='.+',
    sep='_')


# ## Inverting stacked data

# ### How to do it...

# In[23]:


usecol_func = lambda x: 'UGDS_' in x or x == 'INSTNM'
college = pd.read_csv('data/college.csv',
    index_col='INSTNM',
    usecols=usecol_func)
college


# In[24]:


college_stacked = college.stack()
college_stacked


# In[25]:


college_stacked.unstack()


# In[26]:


college2 = pd.read_csv('data/college.csv',
   usecols=usecol_func)
college2


# In[27]:


college_melted = college2.melt(id_vars='INSTNM',
    var_name='Race',
    value_name='Percentage')
college_melted


# In[28]:


melted_inv = college_melted.pivot(index='INSTNM',
    columns='Race',
    values='Percentage')
melted_inv


# In[29]:


college2_replication = (melted_inv
    .loc[college2['INSTNM'], college2.columns[1:]]
    .reset_index()
)
college2.equals(college2_replication)


# ### How it works...

# ### There's more...

# In[30]:


college.stack().unstack(0)


# In[31]:


college.T
college.transpose()


# ## Unstacking after a groupby aggregation

# ### How to do it...

# In[32]:


employee = pd.read_csv('data/employee.csv')
(employee
    .groupby('RACE')
    ['BASE_SALARY']
    .mean()
    .astype(int)
)


# In[33]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
)


# In[34]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('GENDER')
)


# In[35]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('RACE')
)


# ### How it works...

# ### There's more...

# In[36]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
)


# In[37]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
    .unstack('GENDER')
)


# ## Replicating pivot_table with a groupby aggregation

# ### How to do it...

# In[38]:


flights = pd.read_csv('data/flights.csv')
fpt = flights.pivot_table(index='AIRLINE',
    columns='ORG_AIR',
    values='CANCELLED',
    aggfunc='sum',
    fill_value=0).round(2)
fpt


# In[39]:


(flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
)


# In[40]:


fpg = (flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
    .unstack('ORG_AIR', fill_value=0)
)


# In[41]:


fpt.equals(fpg)


# ### How it works...

# ### There's more...

# In[42]:


flights.pivot_table(index=['AIRLINE', 'MONTH'],
    columns=['ORG_AIR', 'CANCELLED'],
    values=['DEP_DELAY', 'DIST'],
    aggfunc=['sum', 'mean'],
    fill_value=0)


# In[43]:


(flights
    .groupby(['AIRLINE', 'MONTH', 'ORG_AIR', 'CANCELLED']) 
    ['DEP_DELAY', 'DIST'] 
    .agg(['mean', 'sum']) 
    .unstack(['ORG_AIR', 'CANCELLED'], fill_value=0) 
    .swaplevel(0, 1, axis='columns')
)


# ## Renaming axis levels for easy reshaping

# ### How to do it...

# In[44]:


college = pd.read_csv('data/college.csv')
(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
)


# In[45]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
)


# In[46]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
)


# In[47]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR',
       axis='index')
)


# In[48]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR', axis='index') 
    .sort_index(level='RELAFFIL', axis='index') 
    .sort_index(level='AGG_COLS', axis='columns')
)


# In[49]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .unstack(['RELAFFIL', 'STABBR'])
)


# In[50]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack(['AGG_FUNCS', 'AGG_COLS'])
)


# In[51]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .unstack(['STABBR', 'RELAFFIL']) 
)


# ### How it works...

# ### There's more...

# In[52]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis([None, None], axis='index') 
    .rename_axis([None, None], axis='columns')
)


# ## Tidying when multiple variables are stored as column names

# ### How to do it...

# In[53]:


weightlifting = pd.read_csv('data/weightlifting_men.csv')
weightlifting


# In[54]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)


# In[55]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
)


# In[56]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
)


# In[57]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0])
)


# In[58]:


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
tidy = pd.concat([melted
           ['sex_age']
           .str.split(expand=True)
           .rename(columns={0:'Sex', 1:'Age Group'})
           .assign(Sex=lambda df_: df_.Sex.str[0]),
          melted[['Weight Category', 'Qual Total']]],
          axis='columns'
)
tidy


# In[59]:


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
(melted
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0],
            Category=melted['Weight Category'],
            Total=melted['Qual Total'])
)


# ### How it works...

# ### There's more...

# In[60]:


tidy2 = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    .assign(Sex=lambda df_:df_.sex_age.str[0],
            **{'Age Group':(lambda df_: (df_
                .sex_age
                .str.extract(r'(\d{2}[-+](?:\d{2})?)',
                             expand=False)))})
    .drop(columns='sex_age')
)


# In[61]:


tidy2


# In[62]:


tidy.sort_index(axis=1).equals(tidy2.sort_index(axis=1))


# ## Tidying when multiple variables are stored is a single column

# ### How to do it...

# In[63]:


inspections = pd.read_csv('data/restaurant_inspections.csv',
    parse_dates=['Date'])
inspections


# In[64]:


inspections.pivot(index=['Name', 'Date'],
    columns='Info', values='Value')


# In[65]:


inspections.set_index(['Name','Date', 'Info'])


# In[66]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
)


# In[67]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
)


# In[68]:


def flatten0(df_):
    df_.columns = df_.columns.droplevel(0).rename(None)
    return df_


# In[69]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
    .pipe(flatten0)
)


# In[70]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .squeeze() 
    .unstack('Info') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ### How it works...

# ### There's more...

# In[71]:


(inspections
    .pivot_table(index=['Name', 'Date'],
                 columns='Info',
                 values='Value',
                 aggfunc='first') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ## Tidying when two or more values are stored in the same cell

# ### How to do it..

# In[72]:


cities = pd.read_csv('data/texas_cities.csv')
cities


# In[73]:


geolocations = cities.Geolocation.str.split(pat='. ',
    expand=True)
geolocations.columns = ['latitude', 'latitude direction',
    'longitude', 'longitude direction']


# In[74]:


geolocations = geolocations.astype({'latitude':'float',
   'longitude':'float'})
geolocations.dtypes


# In[75]:


(geolocations
    .assign(city=cities['City'])
)


# ### How it works...

# In[76]:


geolocations.apply(pd.to_numeric, errors='ignore')


# ### There's more...

# In[77]:


cities.Geolocation.str.split(pat=r'° |, ', expand=True)


# In[78]:


cities.Geolocation.str.extract(r'([0-9.]+). (N|S), ([0-9.]+). (E|W)',
   expand=True)


# ## Tidying when variables are stored in column names and values

# ### Getting ready

# In[79]:


sensors = pd.read_csv('data/sensors.csv')
sensors


# In[80]:


sensors.melt(id_vars=['Group', 'Property'], var_name='Year')


# In[81]:


(sensors
    .melt(id_vars=['Group', 'Property'], var_name='Year') 
    .pivot_table(index=['Group', 'Year'],
                 columns='Property', values='value') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ### How it works...

# ### There's more...

# In[82]:


(sensors
    .set_index(['Group', 'Property']) 
    .stack() 
    .unstack('Property') 
    .rename_axis(['Group', 'Year'], axis='index') 
    .rename_axis(None, axis='columns') 
    .reset_index()
)


# In[ ]:






'''  code11_11장_pandas 객체 병합(join).py  '''

#!/usr/bin/env python
# coding: utf-8

# ## Combining Pandas Objects


import os
os.getcwd()
os.chdir("D:/pandas_cookbook")


# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 7,'display.expand_frame_repr', True, # 'max_rows', 10, 
    'max_colwidth', 9, 'max_rows', 10, #'precision', 2
)#, 'width', 45)
pd.set_option('display.width', 65)


# ## Introduction

# ## Appending new rows to DataFrames

# ### How to do it...

# In[2]:


names = pd.read_csv('data/names.csv')
names


# In[3]:


new_data_list = ['Aria', 1]
names.loc[4] = new_data_list
names


# In[4]:


names.loc['five'] = ['Zach', 3]
names


# In[5]:


names.loc[len(names)] = {'Name':'Zayd', 'Age':2}
names


# In[6]:


names.loc[len(names)] = pd.Series({'Age':32, 'Name':'Dean'})
names


# In[7]:


names = pd.read_csv('data/names.csv')
names.append({'Name':'Aria', 'Age':1})


# In[8]:


names.append({'Name':'Aria', 'Age':1}, ignore_index=True)


# In[9]:


names.index = ['Canada', 'Canada', 'USA', 'USA']
names
 

# In[10]:


s = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s


# In[11]:


names.append(s)


# In[12]:


s1 = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s2 = pd.Series({'Name': 'Zayd', 'Age': 2}, name='USA')
names.append([s1, s2])


# In[13]:


bball_16 = pd.read_csv('data/baseball16.csv')
bball_16


# In[14]:

# 단일 행을 Series로 선택, .to_dict 메서드를 체인시켜 예제 행을 딕셔너리 형태로 가져옴

data_dict = bball_16.iloc[0].to_dict()
data_dict


# In[15]:

# 이전 문자열 값을 모두 빈 문자열로 지정해 지우고, 다른 것은 결측치로 
# 지정하는 딕셔너리 컴프리헨션(dictionary comprehension) 할당

new_data_dict = {k: '' if isinstance(v, str) else
    np.nan for k, v in data_dict.items()}
new_data_dict


# ### How it works...

# ### There's more...

# In[16]:


random_data = []
for i in range(1000):   # doctest: +SKIP
    d = dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            d[k] = np.random.choice(list('abcde'))
        else:
            d[k] = np.random.randint(10)
    random_data.append(pd.Series(d, name=i + len(bball_16)))
random_data[0]


# ## Concatenating multiple DataFrames together

%%timeit 
bball_16_copy = bball_16.copy()
for row in random_data:
    bball_16_copy = bball_16_copy.append(row)



# ### How to do it...

# In[17]:

#  여러 데이터 프레임을 함께 연결
# concat 함수를 사용하면 두 개 이상의 데이터 프레임을 세로와 가로로 함께 연결할 수 있음
'''
평소와 마찬가지로 여러 pandas 객체를 동시에 처리하는 경우 연결은 
우연히 발생하는 것이 아니라 각 객체를 인덱스별로 정렬함
'''

stocks_2016 = pd.read_csv('data/stocks_2016.csv',
    index_col='Symbol')
stocks_2017 = pd.read_csv('data/stocks_2017.csv',
    index_col='Symbol')


# In[18]:


stocks_2016


# In[19]:


stocks_2017


# In[20]:


s_list = [stocks_2016, stocks_2017]
pd.concat(s_list)


# In[21]:


pd.concat(s_list, keys=['2016', '2017'], names=['Year', 'Symbol'])  


# In[22]:


pd.concat(s_list, keys=['2016', '2017'],
    axis='columns', names=['Year', None])    


# In[23]:

# join 방식: 기본 -> outer join
#            'inner' -> inner join

pd.concat(s_list, join='inner', keys=['2016', '2017'],
    axis='columns', names=['Year', None])


# ### How it works...

# ### There's more...

# In[24]:

# .append 함수는 DataFrame에 새 행만 추가할 수 있는 상당히 압축된 버전의 concat
# 내부적으로는 .append는 concat 함수를 호출함

stocks_2016.append(stocks_2017)


# ## Understanding the differences between concat, join, and merge

# ### How to do it...

# In[25]:
'''
concat, join, merge의 차이점 이해
'''

from IPython.display import display_html
years = 2016, 2017, 2018
stock_tables = [pd.read_csv(
    'data/stocks_{}.csv'.format(year), index_col = 'Symbol')
    for year in years]
stocks_2016, stocks_2017, stocks_2018 = stock_tables
stocks_2016


# In[26]:


stocks_2017


# In[27]:


stocks_2018


# In[28]:


pd.concat(stock_tables, keys=[2016, 2017, 2018])


# In[29]:

# axis 매개 변수를 columns로 변경하면 DataFrame을 수평으로 병합할 수 있음
pd.concat(dict(zip(years, stock_tables)), axis='columns')


# In[30]:


stocks_2016.join(stocks_2017, lsuffix='_2016',
    rsuffix='_2017', how='outer')


# In[31]:


other = [stocks_2017.add_suffix('_2017'),
    stocks_2018.add_suffix('_2018')]
stocks_2016.add_suffix('_2016').join(other, how='outer')


# In[32]:


stock_join = stocks_2016.add_suffix('_2016').join(other,
    how='outer')
stock_concat = pd.concat(dict(zip(years,stock_tables)),
    axis='columns')
level_1 = stock_concat.columns.get_level_values(1)
level_0 = stock_concat.columns.get_level_values(0).astype(str)
stock_concat.columns = level_1 + '_' + level_0
stock_join.equals(stock_concat)


# In[33]:


stocks_2016.merge(stocks_2017, left_index=True,
    right_index=True)


# In[34]:


step1 = stocks_2016.merge(stocks_2017, 
                          left_index=True,
                          right_index=True, 
                          how='outer',
                          suffixes=('_2016', '_2017'))

stock_merge = step1.merge(stocks_2018.add_suffix('_2018'),
                          left_index=True, 
                          right_index=True,
                          how='outer')

stock_concat.equals(stock_merge)


# In[35]:

# 인덱스나 열 레이블 자체가 아닌 열 값에 따라 정렬하는 경우 비교

names = ['prices', 'transactions']
food_tables = [pd.read_csv('data/food_{}.csv'.format(name))
    for name in names]
food_prices, food_transactions = food_tables
food_prices


# In[36]:


food_transactions


# In[37]:


food_transactions.merge(food_prices, on=['item', 'store'])    


# In[38]:


food_transactions.merge(food_prices.query('Date == 2017'), how='left')


# In[39]:


food_prices_join = food_prices.query('Date == 2017').set_index(['item', 'store'])
food_prices_join    


# In[40]:


food_transactions.join(food_prices_join, on=['item', 'store'])


# In[41]:


pd.concat([food_transactions.set_index(['item', 'store']),
           food_prices.set_index(['item', 'store'])],
          axis='columns')


# ### How it works...

# ### There's more...

# In[42]:


import glob
df_list = []
for filename in glob.glob('data/gas prices/*.csv'):
    df_list.append(pd.read_csv(filename, 
                               index_col = 'Week',
                               parse_dates = ['Week']))
gas = pd.concat(df_list, axis='columns')
gas


# ## Connecting to SQL databases

# ### How to do it...

# In[43]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')


# In[44]:


tracks = pd.read_sql_table('tracks', engine)
tracks


# In[45]:


(pd.read_sql_table('genres', engine)
     .merge(tracks[['GenreId', 'Milliseconds']],
            on='GenreId', how='left') 
     .drop('GenreId', axis='columns')
)


# In[46]:


(pd.read_sql_table('genres', engine)
     .merge(tracks[['GenreId', 'Milliseconds']],
            on='GenreId', how='left') 
     .drop('GenreId', axis='columns')
     .groupby('Name')
     ['Milliseconds']
     .mean()
     .pipe(lambda s_: pd.to_timedelta(s_, unit='ms'))
     .dt.floor('s')
     .sort_values()
)


# In[47]:


cust = pd.read_sql_table('customers', engine,
    columns=['CustomerId','FirstName',
    'LastName'])
invoice = pd.read_sql_table('invoices', engine,
    columns=['InvoiceId','CustomerId'])
ii = pd.read_sql_table('invoice_items', engine,
    columns=['InvoiceId', 'UnitPrice', 'Quantity'])
(cust
    .merge(invoice, on='CustomerId') 
    .merge(ii, on='InvoiceId')
)


# In[48]:


(cust
    .merge(invoice, on='CustomerId') 
    .merge(ii, on='InvoiceId')
    .assign(Total=lambda df_:df_.Quantity * df_.UnitPrice)
    .groupby(['CustomerId', 'FirstName', 'LastName'])
    ['Total']
    .sum()
    .sort_values(ascending=False) 
)


# ### How it works...

# ### There's more...

# In[49]:


sql_string1 = '''
SELECT
    Name,
    time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
FROM (
      SELECT
          g.Name,
          t.Milliseconds
      FROM
          genres as g
      JOIN
          tracks as t on
          g.genreid == t.genreid
     )
GROUP BY Name
ORDER BY avg_time'''
pd.read_sql_query(sql_string1, engine)


# In[50]:


sql_string2 = '''
   SELECT
         c.customerid,
         c.FirstName,
         c.LastName,
         sum(ii.quantity * ii.unitprice) as Total
   FROM
        customers as c
   JOIN
        invoices as i
        on c.customerid = i.customerid
   JOIN
       invoice_items as ii
       on i.invoiceid = ii.invoiceid
   GROUP BY
       c.customerid, c.FirstName, c.LastName
   ORDER BY
       Total desc'''


# In[51]:


pd.read_sql_query(sql_string2, engine)


# In[ ]:

# SQL 연습

from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')

tracks = pd.read_sql_table('tracks', engine)

(
 pd.read_sql_table('genres', engine)
 .merge(tracks[['GenreId', 'Milliseconds']], on = 'GenreId', how = 'left')
 .drop('GenreId', axis = 'columns')
)


(
 pd.read_sql_table('genres', engine)
 .merge(tracks[['GenreId', 'Milliseconds']], on = 'GenreId', how = 'left')
 .drop('GenreId', axis = 'columns')
 .groupby('Name')
 ['Milliseconds']
 .mean()
 .pipe(lambda s_: pd.to_timedelta(s_, unit = 'ms').rename('Length')
       )
 .dt.floor('s')
 .sort_values()
)

# 고객당 총 지출 추출
cust = pd.read_sql_table('customers', engine,
                         columns = ['CustomerId', 'FirstName', 'LastName'])

cust

invoice = pd.read_sql_table('invoices', engine, 
                            columns = ['InvoiceId', 'CustomerId'])

invoice_items = pd.read_sql_table('invoice_items', 
                                  engine, 
                                  columns = ['InvoiceId', 'UnitPrice', 'Quantity'])

(
 cust.merge(invoice, on = 'CustomerId')
 .merge(invoice_items, on = 'InvoiceId')
)

# 수량과 단위 가격을 곱하면 고객당 총지출을 구할 수 있음
(cust
 .merge(invoice, on = 'CustomerId')
 .merge(invoice_items, on = 'InvoiceId')
 .assign(Total = lambda df_: df_.Quantity*df_.UnitPrice)
 .groupby(['CustomerId', 'FirstName', 'LastName'])
 ['Total']
 .sum()
 .sort_values(ascending = False)
)










'''  code12_12장_시계열 분석(datetime).py  '''

#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis

# In[10]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 7,'display.expand_frame_repr', True, # 'max_rows', 10, 
    'max_colwidth', 12, 'max_rows', 10, #'precision', 2
)#, 'width', 45)
pd.set_option('display.width', 65)


# In[ ]:


pd.set_option(#'max_columns', 4,
    'max_rows', 10)
from io import StringIO
def txt_repr(df, width=40, rows=None):
    buf = StringIO()
    rows = rows if rows is not None else pd.options.display.max_rows
    num_cols = len(df.columns)
    with pd.option_context('display.width', 100):
        df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
        out = buf.getvalue()
        for line in out.split('\n'):
            if len(line) > width or line.strip().endswith('\\'):
                break
        else:
            return out
        done = False
        while not done:
            buf = StringIO()
            df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
            for line in buf.getvalue().split('\n'):
                if line.strip().endswith('\\'):
                    num_cols = min([num_cols - 1, int(num_cols*.8)])
                    break
            else:
                break
        return buf.getvalue()
pd.DataFrame.__repr__ = lambda self, *args: txt_repr(self, 65, 10)


# ## Introduction

# ## Understanding the difference between Python and pandas date tools

# ### How to do it...

# In[11]:


import datetime
date = datetime.date(year=2013, month=6, day=7)
time = datetime.time(hour=12, minute=30,
    second=19, microsecond=463198)
dt = datetime.datetime(year=2013, month=6, day=7,
    hour=12, minute=30, second=19,
    microsecond=463198)
print(f"date is {date}")


# In[12]:


print(f"time is {time}")


# In[13]:


print(f"datetime is {dt}")


# In[14]:


td = datetime.timedelta(weeks=2, days=5, hours=10,
    minutes=20, seconds=6.73,
    milliseconds=99, microseconds=8)
td


# In[15]:


print(f'new date is {date+td}')


# In[16]:


print(f'new datetime is {dt+td}')


# In[17]:


time + td


# In[156]:


pd.Timestamp(year=2012, month=12, day=21, hour=5,
   minute=10, second=8, microsecond=99)


# In[157]:


pd.Timestamp('2016/1/10')


# In[158]:


pd.Timestamp('2014-5/10')


# In[159]:


pd.Timestamp('Jan 3, 2019 20:45.56')


# In[160]:


pd.Timestamp('2016-01-05T05:34:43.123456789')


# In[161]:


pd.Timestamp(500)


# In[162]:


pd.Timestamp(5000, unit='D')


# In[163]:


pd.to_datetime('2015-5-13')


# In[164]:


pd.to_datetime('2015-13-5', dayfirst=True)


# In[165]:


pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm',
    format='Start Date: %b %d, %Y Start Time: %I:%M %p')


# In[166]:


pd.to_datetime(100, unit='D', origin='2013-1-1')


# In[167]:


s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')


# In[168]:


s = pd.Series(['12-5-2015', '14-1-2013',
   '20/12/2017', '40/23/2017'])


# In[169]:


pd.to_datetime(s, dayfirst=True, errors='coerce')


# In[170]:


pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])


# In[171]:


pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')


# In[172]:


pd.Timedelta(days=5, minutes=7.34)


# In[173]:


pd.Timedelta(100, unit='W')


# In[174]:


pd.to_timedelta('67:15:45.454')


# In[175]:


s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')


# In[176]:


time_strings = ['2 days 24 minutes 89.67 seconds',
    '00:45:23.6']
pd.to_timedelta(time_strings)


# In[177]:


pd.Timedelta('12 days 5 hours 3 minutes') * 2


# In[178]:


(pd.Timestamp('1/1/2017') + 
   pd.Timedelta('12 days 5 hours 3 minutes') * 2)


# In[179]:


td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2


# In[180]:


pd.Timedelta('12 days') / pd.Timedelta('3 days')


# In[181]:


ts = pd.Timestamp('2016-10-1 4:23:23.9')
ts.ceil('h')


# In[182]:


ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second


# In[183]:


ts.dayofweek, ts.dayofyear, ts.daysinmonth


# In[184]:


ts.to_pydatetime()


# In[185]:


td = pd.Timedelta(125.8723, unit='h')
td


# In[186]:


td.round('min')


# In[187]:


td.components


# In[188]:


td.total_seconds()


# ### How it works...

# ### There's more...

# In[189]:


date_string_list = ['Sep 30 1984'] * 10000
get_ipython().run_line_magic('timeit', "pd.to_datetime(date_string_list, format='%b %d %Y')")


# In[190]:


get_ipython().run_line_magic('timeit', 'pd.to_datetime(date_string_list)')


# ## Slicing time series intelligently

# ### How to do it...

# In[191]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


# In[192]:


crime = crime.set_index('REPORTED_DATE')
crime


# In[193]:


crime.loc['2016-05-12 16:45:00']


# In[194]:


crime.loc['2016-05-12']


# In[195]:


crime.loc['2016-05'].shape


# In[196]:


crime.loc['2016'].shape


# In[197]:


crime.loc['2016-05-12 03'].shape


# In[198]:


crime.loc['Dec 2015'].sort_index()


# In[199]:


crime.loc['2016 Sep, 15'].shape


# In[200]:


crime.loc['21st October 2014 05'].shape


# In[201]:


crime.loc['2015-3-4':'2016-1-1'].sort_index()


# In[202]:


crime.loc['2015-3-4 22':'2016-1-1 11:22:00'].sort_index()


# ### How it works...

# In[203]:


mem_cat = crime.memory_usage().sum()
mem_obj = (crime
   .astype({'OFFENSE_TYPE_ID':'object',
            'OFFENSE_CATEGORY_ID':'object',
           'NEIGHBORHOOD_ID':'object'}) 
   .memory_usage(deep=True)
   .sum()
)
mb = 2 ** 20
round(mem_cat / mb, 1), round(mem_obj / mb, 1)


# In[204]:


crime.index[:2]


# ### There's more...

# In[205]:


get_ipython().run_line_magic('timeit', "crime.loc['2015-3-4':'2016-1-1']")


# In[206]:


crime_sort = crime.sort_index()
get_ipython().run_line_magic('timeit', "crime_sort.loc['2015-3-4':'2016-1-1']")


# ## Filtering columns with time data

# ### How to do it... 

# In[207]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


# In[208]:


(crime
    [crime.REPORTED_DATE == '2016-05-12 16:45:00']
)


# In[209]:


(crime
    [crime.REPORTED_DATE == '2016-05-12']
)


# In[210]:


(crime
    [crime.REPORTED_DATE.dt.date == '2016-05-12']
)


# In[211]:


(crime
    [crime.REPORTED_DATE.between(
         '2016-05-12', '2016-05-13')]
)


# In[212]:


(crime
    [crime.REPORTED_DATE.between(
         '2016-05', '2016-06')]
    .shape
)


# In[213]:


(crime
    [crime.REPORTED_DATE.between(
         '2016', '2017')]
    .shape
)


# In[214]:


(crime
    [crime.REPORTED_DATE.between(
         '2016-05-12 03', '2016-05-12 04')]
    .shape
)


# In[215]:


(crime
    [crime.REPORTED_DATE.between(
         '2016 Sep, 15', '2016 Sep, 16')]
    .shape
)


# In[216]:


(crime
    [crime.REPORTED_DATE.between(
         '21st October 2014 05', '21st October 2014 06')]
    .shape
)


# In[217]:


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 23:59:59')]
    .shape
)


# In[218]:


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 11:22:00')]
    .shape
)


# ### How it works...

# In[219]:


lmask = crime.REPORTED_DATE >= '2015-3-4 22'
rmask = crime.REPORTED_DATE <= '2016-1-1 11:22:00'
crime[lmask & rmask].shape


# ### There's more...

# In[220]:


ctseries = crime.set_index('REPORTED_DATE')
get_ipython().run_line_magic('timeit', "ctseries.loc['2015-3-4':'2016-1-1']")


# In[221]:


get_ipython().run_line_magic('timeit', "crime[crime.REPORTED_DATE.between('2015-3-4','2016-1-1')]")


# ## Using methods that only work with a DatetimeIndex

# ### How to do it...

# In[222]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE')
)
type(crime.index)


# In[223]:


crime.between_time('2:00', '5:00', include_end=False)


# In[224]:


crime.at_time('5:47')


# In[225]:


crime_sort = crime.sort_index()
crime_sort.first(pd.offsets.MonthBegin(6))


# In[226]:


crime_sort.first(pd.offsets.MonthEnd(6))


# In[227]:


crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))


# In[228]:


crime_sort.loc[:'2012-06']


# In[229]:


crime_sort.first('5D') # 5 days


# In[230]:


crime_sort.first('5B') # 5 business days


# In[231]:


crime_sort.first('7W') # 7 weeks, with weeks ending on Sunday


# In[232]:


crime_sort.first('3QS') # 3rd quarter start


# In[233]:


crime_sort.first('A') # one year end


# ### How it works...

# In[234]:


import datetime
crime.between_time(datetime.time(2,0), datetime.time(5,0),
                   include_end=False)


# In[235]:


first_date = crime_sort.index[0]
first_date


# In[236]:


first_date + pd.offsets.MonthBegin(6)


# In[237]:


first_date + pd.offsets.MonthEnd(6)


# In[238]:


step4 = crime_sort.first(pd.offsets.MonthEnd(6))
end_dt = crime_sort.index[0] + pd.offsets.MonthEnd(6)
step4_internal = crime_sort[:end_dt]
step4.equals(step4_internal)


# ### There's more...

# In[239]:


dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months=1)


# In[240]:


do = pd.DateOffset(years=2, months=5, days=3,
    hours=8, seconds=10)
pd.Timestamp('2012-1-22 03:22') + do


# ## Counting the number of weekly crimes

# ### How to do it...

# In[241]:


crime_sort = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE') 
    .sort_index()
)


# In[242]:


crime_sort.resample('W')


# In[243]:


(crime_sort
    .resample('W')
    .size()
)


# In[244]:


len(crime_sort.loc[:'2012-1-8'])


# In[245]:


len(crime_sort.loc['2012-1-9':'2012-1-15'])


# In[246]:


(crime_sort
    .resample('W-THU')
    .size()
)


# In[247]:


weekly_crimes = (crime_sort
    .groupby(pd.Grouper(freq='W')) 
    .size()
)
weekly_crimes


# ### How it works...

# In[248]:


r = crime_sort.resample('W')
[attr for attr in dir(r) if attr[0].islower()]


# ### There's more...

# In[249]:


crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on='REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)


# In[250]:


weekly_crimes_gby2 = (crime
    .groupby(pd.Grouper(key='REPORTED_DATE', freq='W'))
    .size()
)
weekly_crimes2.equals(weekly_crimes)


# In[251]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 4))
weekly_crimes.plot(title='All Denver Crimes', ax=ax)
fig.savefig('/tmp/c12-crimes.png', dpi=300)


# ## Aggregating weekly crime and traffic accidents separately

# ### How to do it...

# In[252]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE') 
    .sort_index()
)


# In[253]:


(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[254]:


(crime
    .resample('QS')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[255]:


(crime
   .loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']]
   .sum()
)


# In[256]:


(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[257]:


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .plot(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents')
)
fig.savefig('/tmp/c12-crimes2.png', dpi=300)


# ### How it works...

# In[258]:


(crime
    .resample('Q')
    .sum()
)


# In[259]:


(crime_sort
    .resample('QS-MAR')
    ['IS_CRIME', 'IS_TRAFFIC'] 
    .sum()
)


# ### There's more...

# In[260]:


crime_begin = (crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .iloc[0]
)


# In[261]:


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .div(crime_begin)
    .sub(1)
    .round(2)
    .mul(100)
    .plot.bar(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents % Increase')
)


# In[262]:


fig.autofmt_xdate()
fig.savefig('/tmp/c12-crimes3.png', dpi=300, bbox_inches='tight')


# ## Measuring crime by weekday and year

# ### How to do it...

# In[263]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime


# In[264]:


(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
)


# In[265]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
fig, ax = plt.subplots(figsize=(6, 4))
(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
   .reindex(days)
   .plot.barh(title=title, ax=ax)
)
fig.savefig('/tmp/c12-crimes4.png', dpi=300, bbox_inches='tight')                 


# In[266]:


title = 'Denver Crimes and Traffic Accidents per Year'
fig, ax = plt.subplots(figsize=(6, 4))
(crime
   ['REPORTED_DATE']
   .dt.year 
   .value_counts()
   .plot.barh(title=title, ax=ax)
)
fig.savefig('/tmp/c12-crimes5.png', dpi=300, bbox_inches='tight')                 


# In[267]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
)


# In[268]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
)


# In[269]:


criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()


# In[270]:


round(272 / 365, 3)


# In[271]:


crime_pct = (crime
   ['REPORTED_DATE']
   .dt.dayofyear.le(272) 
   .groupby(crime.REPORTED_DATE.dt.year) 
   .mean()
   .round(3)
)


# In[272]:


crime_pct


# In[273]:


crime_pct.loc[2012:2016].median()


# In[274]:


def update_2017(df_):
    df_.loc[2017] = (df_
        .loc[2017]
        .div(.748) 
        .astype('int')
    )
    return df_
(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)


# In[275]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
table = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)
sns.heatmap(table, cmap='Greys', ax=ax)
fig.savefig('/tmp/c12-crimes6.png', dpi=300, bbox_inches='tight')                 


# In[276]:


denver_pop = pd.read_csv('data/denver_pop.csv',
    index_col='Year')
denver_pop


# In[277]:


den_100k = denver_pop.div(100_000).squeeze()
normalized = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# In[278]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(normalized, cmap='Greys', ax=ax)
fig.savefig('/tmp/c12-crimes7.png', dpi=300, bbox_inches='tight')                 


# ### How it works...

# In[279]:


(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
   .loc[days]
)


# In[280]:


(crime
    .assign(year=crime.REPORTED_DATE.dt.year,
            day=crime.REPORTED_DATE.dt.weekday_name)
    .pipe(lambda df_: pd.crosstab(df_.year, df_.day))
)


# In[281]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
) / den_100k


# ### There's more...

# In[282]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
crime_type = 'auto-theft'
normalized = (crime
    .query('OFFENSE_CATEGORY_ID == @crime_type')
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# ## Grouping with anonymous functions with a DatetimeIndex

# ### How to do it...

# In[283]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
   .set_index('REPORTED_DATE') 
   .sort_index()
)


# In[284]:


common_attrs = (set(dir(crime.index)) & 
    set(dir(pd.Timestamp)))
[attr for attr in common_attrs if attr[0] != '_']


# In[285]:


crime.index.weekday_name.value_counts()


# In[286]:


(crime
   .groupby(lambda idx: idx.weekday_name) 
   ['IS_CRIME', 'IS_TRAFFIC']
   .sum()    
)


# In[287]:


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
)


# In[288]:


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
    .style.highlight_max(color='lightgrey')
)


# ### How it works...

# ## Grouping by a Timestamp and another column

# ### How to do it...

# In[289]:


employee = pd.read_csv('data/employee.csv',
    parse_dates=['JOB_DATE', 'HIRE_DATE'],
    index_col='HIRE_DATE')
employee


# In[290]:


(employee
    .groupby('GENDER')
    ['BASE_SALARY']
    .mean()
    .round(-2)
)


# In[291]:


(employee
    .resample('10AS')
    ['BASE_SALARY']
    .mean()
    .round(-2)    
)


# In[292]:


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
)


# In[293]:


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# In[294]:


employee[employee['GENDER'] == 'Male'].index.min()


# In[295]:


employee[employee['GENDER'] == 'Female'].index.min()


# In[296]:


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
)


# In[297]:


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# ### How it works...

# ### There's more...

# In[298]:


sal_final = (employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)
years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final


# In[299]:


cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts.categories.values


# In[300]:


(employee
    .groupby([cuts, 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .unstack('GENDER')
    .round(-2)
)


# In[ ]:






'''  code13_13장_matplotlib, pandas, seaborn을 이용한 시각화.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Visualization with Matplotlib, Pandas, and Seaborn

# ## Introduction

# ## Getting started with matplotlib

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object-oriented guide to matplotlib

# In[21]:


import matplotlib.pyplot as plt
x = [-3, 5, 7]
y = [10, 2, 5]
fig = plt.figure(figsize=(15,3))
plt.plot(x, y)
plt.xlim(0, 10)
plt.ylim(-3, 8)
plt.xlabel('X Axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Figure Title', size=20, y=1.03)
fig.savefig('c13-fig1.png', dpi=300, bbox_inches='tight')


# In[22]:


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.core.display import display
fig = Figure(figsize=(15, 3))
FigureCanvas(fig)  
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)
display(fig)
fig.savefig('c13-fig2.png', dpi=300, bbox_inches='tight')


# In[23]:


fig, ax = plt.subplots(figsize=(15,3))
ax.plot(x, y)
ax.set(xlim=(0, 10), ylim=(-3, 8),
    xlabel='X axis', ylabel='Y axis',
    title='Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)
fig.savefig('c13-fig3.png', dpi=300, bbox_inches='tight')


# ### How to do it...

# In[24]:


import matplotlib.pyplot as plt


# In[25]:


fig, ax = plt.subplots(nrows=1, ncols=1)
fig.savefig('c13-step2.png', dpi=300)         


# In[26]:


type(fig)


# In[27]:


type(ax)


# In[28]:


fig.get_size_inches()


# In[29]:


fig.set_size_inches(14, 4)
fig.savefig('c13-step4.png', dpi=300)         
fig


# In[30]:


fig.axes


# In[31]:


fig.axes[0] is ax


# In[32]:


fig.set_facecolor('.7')
ax.set_facecolor('.5')
fig.savefig('c13-step7.png', dpi=300, facecolor='.7')  
fig


# In[33]:


ax_children = ax.get_children()
ax_children


# In[34]:


spines = ax.spines
spines


# In[35]:


spine_left = spines['left']
spine_left.set_position(('outward', -100))
spine_left.set_linewidth(5)
spine_bottom = spines['bottom']
spine_bottom.set_visible(False)
fig.savefig('c13-step10.png', dpi=300, facecolor='.7')
fig


# In[36]:


ax.xaxis.grid(True, which='major', linewidth=2,
    color='black', linestyle='--')
ax.xaxis.set_ticks([.2, .4, .55, .93])
ax.xaxis.set_label_text('X Axis', family='Verdana',
    fontsize=15)
ax.set_ylabel('Y Axis', family='Gotham', fontsize=20)
ax.set_yticks([.1, .9])
ax.set_yticklabels(['point 1', 'point 9'], rotation=45)
fig.savefig('c13-step11.png', dpi=300, facecolor='.7')         


# ### How it works...

# In[37]:


plot_objects = plt.subplots(nrows=1, ncols=1)
type(plot_objects)


# In[38]:


fig = plot_objects[0]
ax = plot_objects[1]
fig.savefig('c13-1-works1.png', dpi=300)         


# In[39]:


fig, axs = plt.subplots(2, 4)
fig.savefig('c13-1-works2.png', dpi=300)         


# In[40]:


axs


# In[41]:


ax = axs[0][0]
fig.axes == fig.get_axes()


# In[42]:


ax.xaxis == ax.get_xaxis()


# In[43]:


ax.yaxis == ax.get_yaxis()


# ### There's more...

# In[44]:


ax.xaxis.properties()


# ## Visualizing data with matplotlib

# ### How to do it...

# In[45]:


import pandas as pd
import numpy as np
alta = pd.read_csv('data/alta-noaa-1980-2019.csv')
alta


# In[46]:


data = (alta
    .assign(DATE=pd.to_datetime(alta.DATE))
    .set_index('DATE')
    .loc['2018-09':'2019-08']
    .SNWD
)
data


# In[47]:


blue = '#99ddee'
white = '#ffffff'
fig, ax = plt.subplots(figsize=(12,4), 
     linewidth=5, facecolor=blue)
ax.set_facecolor(blue)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', colors=white)
ax.tick_params(axis='y', colors=white)
ax.set_ylabel('Snow Depth (in)', color=white)
ax.set_title('2009-2010', color=white, fontweight='bold')
ax.fill_between(data.index, data, color=white)
fig.savefig('c13-alta1.png', dpi=300, facecolor=blue)  


# In[48]:


import matplotlib.dates as mdt
blue = '#99ddee'
white = '#ffffff'


# In[49]:


def plot_year(ax, data, years):
    ax.set_facecolor(blue)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors=white)
    ax.tick_params(axis='y', colors=white)
    ax.set_ylabel('Snow Depth (in)', color=white)
    ax.set_title(years, color=white, fontweight='bold')
    ax.fill_between(data.index, data, color=white)


# In[50]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
    figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta
       .assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)


# In[51]:


fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('c13-alta2.png', dpi=300, facecolor=blue)  


# ### How it works...

# ### There's more...

# In[121]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
    figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
       .interpolate()
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)
plt.tight_layout()


# In[53]:


fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('c13-alta3.png', dpi=300, facecolor=blue)  


# In[54]:


(alta
    .assign(DATE=pd.to_datetime(alta.DATE))
    .set_index('DATE')
    .SNWD
    .to_frame()
    .assign(next=lambda df_:df_.SNWD.shift(-1),
            snwd_diff=lambda df_:df_.next-df_.SNWD)
    .pipe(lambda df_: df_[df_.snwd_diff.abs() > 50])
)


# In[55]:


def fix_gaps(ser, threshold=50):
    'Replace values where the shift is > threshold with nan'
    mask = (ser
       .to_frame()
       .assign(next=lambda df_:df_.SNWD.shift(-1),
               snwd_diff=lambda df_:df_.next-df_.SNWD)
       .pipe(lambda df_: df_.snwd_diff.abs() > threshold)
    )
    return ser.where(~mask, np.nan)


# In[56]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
    figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
       .pipe(fix_gaps)
       .interpolate()
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)


# In[57]:


fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('c13-alta4.png', dpi=300, facecolor=blue)  


# ## Plotting basics with pandas

# ### How to do it..

# In[58]:


df = pd.DataFrame(index=['Atiya', 'Abbas', 'Cornelia',
    'Stephanie', 'Monte'],
    data={'Apples':[20, 10, 40, 20, 50],
          'Oranges':[35, 40, 25, 19, 33]})


# In[59]:


df


# In[60]:


color = ['.2', '.7']
ax = df.plot.bar(color=color, figsize=(16,4))
ax.get_figure().savefig('c13-pdemo-bar1.png', dpi=300, bbox_inches='tight')


# In[61]:


ax = df.plot.kde(color=color, figsize=(16,4))
ax.get_figure().savefig('c13-pdemo-kde1.png')


# In[62]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
fig.suptitle('Two Variable Plots', size=20, y=1.02)
df.plot.line(ax=ax1, title='Line plot')
df.plot.scatter(x='Apples', y='Oranges', 
    ax=ax2, title='Scatterplot')
df.plot.bar(color=color, ax=ax3, title='Bar plot')
fig.savefig('c13-pdemo-scat.png', dpi=300, bbox_inches='tight')


# In[63]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
fig.suptitle('One Variable Plots', size=20, y=1.02)
df.plot.kde(color=color, ax=ax1, title='KDE plot')
df.plot.box(ax=ax2, title='Boxplot')
df.plot.hist(color=color, ax=ax3, title='Histogram')
fig.savefig('c13-pdemo-kde2.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ### There's more...

# In[64]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
df.sort_values('Apples').plot.line(x='Apples', y='Oranges',
      ax=ax1)
df.plot.bar(x='Apples', y='Oranges', ax=ax2)
df.plot.kde(x='Apples', ax=ax3)
fig.savefig('c13-pdemo-kde3.png', dpi=300, bbox_inches='tight')


# ## Visualizing the flights dataset

# ### How to do it...

# In[65]:


flights = pd.read_csv('data/flights.csv')
flights


# In[66]:


cols = ['DIVERTED', 'CANCELLED', 'DELAYED']
(flights
    .assign(DELAYED=flights['ARR_DELAY'].ge(15).astype(int),
            ON_TIME=lambda df_:1 - df_[cols].any(axis=1))
    .select_dtypes(int)
    .sum()
)


# In[67]:


fig, ax_array = plt.subplots(2, 3, figsize=(18,8))
(ax1, ax2, ax3), (ax4, ax5, ax6) = ax_array
fig.suptitle('2015 US Flights - Univariate Summary', size=20)
ac = flights['AIRLINE'].value_counts()
ac.plot.barh(ax=ax1, title='Airline')
(flights
    ['ORG_AIR']
    .value_counts()
    .plot.bar(ax=ax2, rot=0, title='Origin City')
)
(flights
    ['DEST_AIR']
    .value_counts()
    .head(10)
    .plot.bar(ax=ax3, rot=0, title='Destination City')
)
(flights
    .assign(DELAYED=flights['ARR_DELAY'].ge(15).astype(int),
            ON_TIME=lambda df_:1 - df_[cols].any(axis=1))
    [['DIVERTED', 'CANCELLED', 'DELAYED', 'ON_TIME']]
    .sum()
    .plot.bar(ax=ax4, rot=0,
         log=True, title='Flight Status')
)
flights['DIST'].plot.kde(ax=ax5, xlim=(0, 3000),
    title='Distance KDE')
flights['ARR_DELAY'].plot.hist(ax=ax6,
    title='Arrival Delay',
    range=(0,200)
)
fig.savefig('c13-uni1.png')


# In[68]:


df_date = (flights
    [['MONTH', 'DAY']]
    .assign(YEAR=2015,
            HOUR=flights['SCHED_DEP'] // 100,
            MINUTE=flights['SCHED_DEP'] % 100)
)
df_date


# In[69]:


flight_dep = pd.to_datetime(df_date)
flight_dep


# In[70]:


flights.index = flight_dep
fc = flights.resample('W').size()
fc.plot.line(figsize=(12,6), title='Flights per Week', grid=True)
fig.savefig('c13-ts1.png', dpi=300, bbox_inches='tight')


# In[71]:


def interp_lt_n(df_, n=600):
    return (df_
        .where(df_ > n)
        .interpolate(limit_direction='both')
)
fig, ax = plt.subplots(figsize=(16,4))
data = (flights
    .resample('W')
    .size()
)
(data
    .pipe(interp_lt_n)
    .iloc[1:-1]
    .plot.line(color='black', ax=ax)
)
mask = data<600
(data
     .pipe(interp_lt_n)
     [mask]
     .plot.line(color='.8', linewidth=10)
) 
ax.annotate(xy=(.8, .55), xytext=(.8, .77),
            xycoords='axes fraction', s='missing data',
            ha='center', size=20, arrowprops=dict())
ax.set_title('Flights per Week (Interpolated Missing Data)')
fig.savefig('c13-ts2.png')


# In[72]:


fig, ax = plt.subplots(figsize=(16,4))
(flights
    .groupby('DEST_AIR')
    ['DIST'] 
    .agg(['mean', 'count']) 
    .query('count > 100') 
    .sort_values('mean') 
    .tail(10) 
    .plot.bar(y='mean', rot=0, legend=False, ax=ax,
        title='Average Distance per Destination')
)
fig.savefig('c13-bar1.png')


# In[73]:


fig, ax = plt.subplots(figsize=(8,6))
(flights
    .reset_index(drop=True)
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .plot.scatter(x='DIST', y='AIR_TIME', ax=ax, alpha=.1, s=1)
)
fig.savefig('c13-scat1.png')


# In[74]:


(flights
    .reset_index(drop=True)
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .pipe(lambda df_:pd.cut(df_.DIST,
          bins=range(0, 2001, 250)))
    .value_counts()
    .sort_index()
)


# In[75]:


zscore = lambda x: (x - x.mean()) / x.std()
short = (flights
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .reset_index(drop=True)    
    .assign(BIN=lambda df_:pd.cut(df_.DIST,
        bins=range(0, 2001, 250)))
)


# In[76]:


scores = (short
    .groupby('BIN')
    ['AIR_TIME']
    .transform(zscore)
)  


# In[77]:


(short.assign(SCORE=scores))


# In[78]:


fig, ax = plt.subplots(figsize=(10,6))    
(short.assign(SCORE=scores)
    .pivot(columns='BIN')
    ['SCORE']
    .plot.box(ax=ax)
)
ax.set_title('Z-Scores for Distance Groups')
fig.savefig('c13-box2.png')


# In[79]:


mask = (short
    .assign(SCORE=scores)
    .pipe(lambda df_:df_.SCORE.abs() >6)
)


# In[80]:


outliers = (flights
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .reset_index(drop=True)
    [mask]
    .assign(PLOT_NUM=lambda df_:range(1, len(df_)+1))
)


# In[81]:


outliers


# In[82]:


fig, ax = plt.subplots(figsize=(8,6))
(short
    .assign(SCORE=scores)
    .plot.scatter(x='DIST', y='AIR_TIME',
                  alpha=.1, s=1, ax=ax,
                  table=outliers)
)
outliers.plot.scatter(x='DIST', y='AIR_TIME',
    s=25, ax=ax, grid=True)
outs = outliers[['AIR_TIME', 'DIST', 'PLOT_NUM']]
for t, d, n in outs.itertuples(index=False):
    ax.text(d + 5, t + 5, str(n))
plt.setp(ax.get_xticklabels(), y=.1)
plt.setp(ax.get_xticklines(), visible=False)
ax.set_xlabel('')
ax.set_title('Flight Time vs Distance with Outliers')
fig.savefig('c13-scat3.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ## Stacking area charts to discover emerging trends

# ### How to do it...

# In[83]:


meetup = pd.read_csv('data/meetup_groups.csv',
    parse_dates=['join_date'],
    index_col='join_date')
meetup


# In[84]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
)


# In[85]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
)


# In[86]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
)


# In[87]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
    .pipe(lambda df_: df_.div(
          df_.sum(axis='columns'), axis='index'))
)


# In[88]:


fig, ax = plt.subplots(figsize=(18,6))    
(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
    .pipe(lambda df_: df_.div(
          df_.sum(axis='columns'), axis='index'))
    .plot.area(ax=ax,
          cmap='Greys', xlim=('2013-6', None),
          ylim=(0, 1), legend=False)
)
ax.figure.suptitle('Houston Meetup Groups', size=25)
ax.set_xlabel('')
ax.yaxis.tick_right()
kwargs = {'xycoords':'axes fraction', 'size':15}
ax.annotate(xy=(.1, .7), s='R Users',
    color='w', **kwargs)
ax.annotate(xy=(.25, .16), s='Data Visualization',
    color='k', **kwargs)
ax.annotate(xy=(.5, .55), s='Energy Data Science',
    color='k', **kwargs)
ax.annotate(xy=(.83, .07), s='Data Science',
    color='k', **kwargs)
ax.annotate(xy=(.86, .78), s='Machine Learning',
    color='w', **kwargs)
fig.savefig('c13-stacked1.png')


# ### How it works...

# ## Understanding the differences between seaborn and pandas

# ### How to do it...

# In[89]:


employee = pd.read_csv('data/employee.csv',
    parse_dates=['HIRE_DATE', 'JOB_DATE'])
employee


# In[90]:


import seaborn as sns


# In[91]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y='DEPARTMENT', data=employee, ax=ax)     
fig.savefig('c13-sns1.png', dpi=300, bbox_inches='tight')


# In[92]:


fig, ax = plt.subplots(figsize=(8, 6))
(employee
    ['DEPARTMENT']
    .value_counts()
    .plot.barh(ax=ax)
)
fig.savefig('c13-sns2.png', dpi=300, bbox_inches='tight')


# In[93]:


fig, ax = plt.subplots(figsize=(8, 6))    
sns.barplot(y='RACE', x='BASE_SALARY', data=employee, ax=ax)
fig.savefig('c13-sns3.png', dpi=300, bbox_inches='tight')


# In[94]:


fig, ax = plt.subplots(figsize=(8, 6))    
(employee
    .groupby('RACE', sort=False) 
    ['BASE_SALARY']
    .mean()
    .plot.barh(rot=0, width=.8, ax=ax)
)
ax.set_xlabel('Mean Salary')
fig.savefig('c13-sns4.png', dpi=300, bbox_inches='tight')


# In[95]:


fig, ax = plt.subplots(figsize=(18, 6))        
sns.barplot(x='RACE', y='BASE_SALARY', hue='GENDER',
    ax=ax, data=employee, palette='Greys',
    order=['Hispanic/Latino', 
           'Black or African American',
           'American Indian or Alaskan Native',
           'Asian/Pacific Islander', 'Others',
           'White'])
fig.savefig('c13-sns5.png', dpi=300, bbox_inches='tight')


# In[96]:


fig, ax = plt.subplots(figsize=(18, 6))            
(employee
    .groupby(['RACE', 'GENDER'], sort=False) 
    ['BASE_SALARY']
    .mean()
    .unstack('GENDER')
    .sort_values('Female')
    .plot.bar(rot=0, ax=ax,
        width=.8, cmap='viridis')
)
fig.savefig('c13-sns6.png', dpi=300, bbox_inches='tight')


# In[97]:


fig, ax = plt.subplots(figsize=(8, 6))            
sns.boxplot(x='GENDER', y='BASE_SALARY', data=employee,
            hue='RACE', palette='Greys', ax=ax)
fig.savefig('c13-sns7.png', dpi=300, bbox_inches='tight')


# In[98]:


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for g, ax in zip(['Female', 'Male'], axs):
    (employee
        .query('GENDER == @g')
        .assign(RACE=lambda df_:df_.RACE.fillna('NA'))
        .pivot(columns='RACE')
        ['BASE_SALARY']
        .plot.box(ax=ax, rot=30)
    )
    ax.set_title(g + ' Salary')
    ax.set_xlabel('')
fig.savefig('c13-sns8.png', bbox_inches='tight')


# ### How it works...

# ## Multivariate analysis with seaborn Grids

# ### How to do it...

# In[99]:


emp = pd.read_csv('data/employee.csv',
    parse_dates=['HIRE_DATE', 'JOB_DATE'])
def yrs_exp(df_):
    days_hired = pd.to_datetime('12-1-2016') - df_.HIRE_DATE
    return days_hired.dt.days / 365.25


# In[100]:


emp = (emp
    .assign(YEARS_EXPERIENCE=yrs_exp)
)


# In[101]:


emp[['HIRE_DATE', 'YEARS_EXPERIENCE']]


# In[102]:


fig, ax = plt.subplots(figsize=(8, 6))        
sns.regplot(x='YEARS_EXPERIENCE', y='BASE_SALARY',
    data=emp, ax=ax)
fig.savefig('c13-scat4.png', dpi=300, bbox_inches='tight')


# In[103]:


grid = sns.lmplot(x='YEARS_EXPERIENCE', y='BASE_SALARY', 
    hue='GENDER', palette='Greys',
    scatter_kws={'s':10}, data=emp)
grid.fig.set_size_inches(8, 6) 
grid.fig.savefig('c13-scat5.png', dpi=300, bbox_inches='tight')


# In[104]:


grid = sns.lmplot(x='YEARS_EXPERIENCE', y='BASE_SALARY',
                  hue='GENDER', col='RACE', col_wrap=3,
                  palette='Greys', sharex=False,
                  line_kws = {'linewidth':5},
                  data=emp)
grid.set(ylim=(20000, 120000))     
grid.fig.savefig('c13-scat6.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ### There's more...

# In[105]:


deps = emp['DEPARTMENT'].value_counts().index[:2]
races = emp['RACE'].value_counts().index[:3]
is_dep = emp['DEPARTMENT'].isin(deps)
is_race = emp['RACE'].isin(races)    
emp2 = (emp
    [is_dep & is_race]
    .assign(DEPARTMENT=lambda df_:
            df_['DEPARTMENT'].str.extract('(HPD|HFD)',
                                    expand=True))
)


# In[106]:


emp2.shape


# In[107]:


emp2['DEPARTMENT'].value_counts()


# In[108]:


emp2['RACE'].value_counts()


# In[109]:


common_depts = (emp
    .groupby('DEPARTMENT') 
    .filter(lambda group: len(group) > 50)
)


# In[110]:


fig, ax = plt.subplots(figsize=(8, 6))   
sns.violinplot(x='YEARS_EXPERIENCE', y='GENDER',
    data=common_depts)
fig.savefig('c13-vio1.png', dpi=300, bbox_inches='tight')


# In[111]:


grid = sns.catplot(x='YEARS_EXPERIENCE', y='GENDER',
                      col='RACE', row='DEPARTMENT',
                      height=3, aspect=2,
                      data=emp2, kind='violin')
grid.fig.savefig('c13-vio2.png', dpi=300, bbox_inches='tight')


# ## Uncovering Simpson's Paradox in the diamonds dataset with seaborn

# ### How to do it...

# In[112]:


dia = pd.read_csv('data/diamonds.csv')
dia


# In[113]:


cut_cats = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_cats = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_cats = ['I1', 'SI2', 'SI1', 'VS2',
                'VS1', 'VVS2', 'VVS1', 'IF']
dia2 = (dia
    .assign(cut=pd.Categorical(dia['cut'], 
                 categories=cut_cats,
                 ordered=True),
            color=pd.Categorical(dia['color'], 
                 categories=color_cats,
                 ordered=True),
            clarity=pd.Categorical(dia['clarity'], 
                 categories=clarity_cats,
                 ordered=True))
)


# In[114]:


dia2


# In[115]:


import seaborn as sns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
sns.barplot(x='color', y='price', data=dia2, ax=ax1)
sns.barplot(x='cut', y='price', data=dia2, ax=ax2)
sns.barplot(x='clarity', y='price', data=dia2, ax=ax3)
fig.suptitle('Price Decreasing with Increasing Quality?')
fig.savefig('c13-bar4.png', dpi=300, bbox_inches='tight')


# In[116]:


grid = sns.catplot(x='color', y='price', col='clarity',
    col_wrap=4, data=dia2, kind='bar')
grid.fig.savefig('c13-bar5.png', dpi=300, bbox_inches='tight')


# In[117]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
sns.barplot(x='color', y='carat', data=dia2, ax=ax1)
sns.barplot(x='cut', y='carat', data=dia2, ax=ax2)
sns.barplot(x='clarity', y='carat', data=dia2, ax=ax3)
fig.suptitle('Diamond size decreases with quality')
fig.savefig('c13-bar6.png', dpi=300, bbox_inches='tight')


# In[118]:


dia2 = (dia2
    .assign(carat_category=pd.qcut(dia2.carat, 5))
)


# In[119]:


from matplotlib.cm import Greys
greys = Greys(np.arange(50,250,40))
grid = sns.catplot(x='clarity', y='price', data=dia2,
   hue='carat_category', col='color',
   col_wrap=4, kind='point', palette=greys)
grid.fig.suptitle('Diamond price by size, color and clarity',
   y=1.02, size=20)
grid.fig.savefig('c13-bar7.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ### There's more...

# In[122]:


g = sns.PairGrid(dia2, height=5,
    x_vars=["color", "cut", "clarity"],
    y_vars=["price"])
g.map(sns.barplot)
g.fig.suptitle('Replication of Step 3 with PairGrid', y=1.02)
g.fig.savefig('c13-bar8.png', dpi=300, bbox_inches='tight')


# In[ ]:






'''  code14_14장_pandas 디버깅과 테스트.py  '''

#!/usr/bin/env python
# coding: utf-8

# # Debugging and Testing Pandas

# ## Code to Transform Data

# ### How to do it...

# In[1]:


import pandas as pd
import numpy as np
import zipfile
url = 'data/kaggle-survey-2018.zip'


# In[2]:


with zipfile.ZipFile(url) as z:
    print(z.namelist())
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


# In[3]:


df.T


# In[4]:


df.dtypes


# In[5]:


df.Q1.value_counts(dropna=False)


# In[6]:


def tweak_kag(df):
    na_mask = df.Q9.isna()
    hide_mask = df.Q9.str.startswith('I do not').fillna(False)
    df = df[~na_mask & ~hide_mask]


    q1 = (df.Q1
      .replace({'Prefer not to say': 'Another',
               'Prefer to self-describe': 'Another'})
      .rename('Gender')
    )
    q2 = df.Q2.str.slice(0,2).astype(int).rename('Age')
    def limit_countries(val):
        if val in  {'United States of America', 'India', 'China'}:
            return val
        return 'Another'
    q3 = df.Q3.apply(limit_countries).rename('Country')


    q4 = (df.Q4
     .replace({'Master’s degree': 18,
     'Bachelor’s degree': 16,
     'Doctoral degree': 20,
     'Some college/university study without earning a bachelor’s degree': 13,
     'Professional degree': 19,
     'I prefer not to answer': None,
     'No formal education past high school': 12})
     .fillna(11)
     .rename('Edu')
    )


    def only_cs_stat_val(val):
        if val not in {'cs', 'eng', 'stat'}:
            return 'another'
        return val


    q5 = (df.Q5
            .replace({
                'Computer science (software engineering, etc.)': 'cs',
                'Engineering (non-computer focused)': 'eng',
                'Mathematics or statistics': 'stat'})
             .apply(only_cs_stat_val)
             .rename('Studies'))
    def limit_occupation(val):
        if val in {'Student', 'Data Scientist', 'Software Engineer', 'Not employed',
                  'Data Engineer'}:
            return val
        return 'Another'


    q6 = df.Q6.apply(limit_occupation).rename('Occupation')


    q8 = (df.Q8
      .str.replace('+', '')
      .str.split('-', expand=True)
      .iloc[:,0]
      .fillna(-1)
      .astype(int)
      .rename('Experience')
    )


    q9 = (df.Q9
     .str.replace('+','')
     .str.replace(',','')
     .str.replace('500000', '500')
     .str.replace('I do not wish to disclose my approximate yearly compensation','')
     .str.split('-', expand=True)
     .iloc[:,0]
     .astype(int)
     .mul(1000)
     .rename('Salary'))
    return pd.concat([q1, q2, q3, q4, q5, q6, q8, q9], axis=1)


# In[7]:


tweak_kag(df)


# In[8]:


tweak_kag(df).dtypes


# ### How it works...

# In[9]:


kag = tweak_kag(df)
(kag
    .groupby('Country')
    .apply(lambda g: g.Salary.corr(g.Experience))
)


# ## Apply Performance

# ### How to do it...

# In[13]:


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# In[14]:


get_ipython().run_cell_magic('timeit', '', "q3 = df.Q3.apply(limit_countries).rename('Country')")


# In[15]:


get_ipython().run_cell_magic('timeit', '', "other_values = df.Q3.value_counts().iloc[3:].index\nq3_2 = df.Q3.replace(other_values, 'Another')")


# In[16]:


get_ipython().run_cell_magic('timeit', '', "values = {'United States of America', 'India', 'China'}\nq3_3 = df.Q3.where(df.Q3.isin(values), 'Another')")


# In[17]:


get_ipython().run_cell_magic('timeit', '', "values = {'United States of America', 'India', 'China'}\nq3_4 = pd.Series(np.where(df.Q3.isin(values), df.Q3, 'Another'), \n     index=df.index)")


# In[18]:


q3.equals(q3_2)


# In[ ]:


q3.equals(q3_3)


# In[ ]:


q3.equals(q3_4)


# ### How it works...

# ### There's more...

# In[19]:


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# In[20]:


q3 = df.Q3.apply(limit_countries).rename('Country')


# In[21]:


def debug(something):
    # what is something? A cell, series, dataframe?
    print(type(something), something)
    1/0


# In[22]:


q3.apply(debug)


# In[28]:


the_item = None
def debug(something):
    global the_item
    the_item = something
    return something


# In[29]:


_ = q3.apply(debug)


# In[30]:


the_item


# ## Improving Apply Performance with Dask, Pandarell, Swifter, and More

# ### How to do it...

# In[31]:


from pandarallel import pandarallel
pandarallel.initialize()


# In[32]:


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# In[33]:


get_ipython().run_cell_magic('timeit', '', "res_p = df.Q3.parallel_apply(limit_countries).rename('Country')")


# In[41]:


import swifter


# In[42]:


get_ipython().run_cell_magic('timeit', '', "res_s = df.Q3.swifter.apply(limit_countries).rename('Country')")


# In[43]:


import dask


# In[44]:


get_ipython().run_cell_magic('timeit', '', "res_d = (dask.dataframe.from_pandas(\n       df, npartitions=4)\n   .map_partitions(lambda df: df.Q3.apply(limit_countries))\n   .rename('Countries')\n)")


# In[45]:


np_fn = np.vectorize(limit_countries)


# In[39]:


get_ipython().run_cell_magic('timeit', '', "res_v = df.Q3.apply(np_fn).rename('Country')")


# In[46]:


from numba import jit


# In[50]:


@jit
def limit_countries2(val):
     if val in  ['United States of America', 'India', 'China']:
         return val
     return 'Another'


# In[51]:


get_ipython().run_cell_magic('timeit', '', "res_n = df.Q3.apply(limit_countries2).rename('Country')")


# ### How it works...

# ## Inspecting Code 

# ### How to do it...

# In[52]:


import zipfile
url = 'data/kaggle-survey-2018.zip'


# In[53]:


with zipfile.ZipFile(url) as z:
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


# In[54]:


get_ipython().run_line_magic('pinfo', 'df.Q3.apply')


# In[55]:


get_ipython().run_line_magic('pinfo2', 'df.Q3.apply')


# In[56]:


import pandas.core.series
pandas.core.series.lib


# In[57]:


get_ipython().run_line_magic('pinfo2', 'pandas.core.series.lib.map_infer')


# ### How it works...

# ### There's more...

# ## Debugging in Jupyter

# ### How to do it...

# In[58]:


import zipfile
url = 'data/kaggle-survey-2018.zip'


# In[59]:


with zipfile.ZipFile(url) as z:
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


# In[60]:


def add1(x):
    return x + 1


# In[61]:


df.Q3.apply(add1)


# In[62]:


from IPython.core.debugger import set_trace


# In[63]:


def add1(x):
    set_trace()
    return x + 1


# In[ ]:


df.Q3.apply(add1)


# ### How it works...

# ### There's more...

# ##  Managing data integrity with Great Expectations

# ### How to do it...

# In[64]:


kag = tweak_kag(df)


# In[66]:


import great_expectations as ge
kag_ge = ge.from_pandas(kag)


# In[67]:


sorted([x for x in set(dir(kag_ge)) - set(dir(kag))
    if not x.startswith('_')])


# In[68]:


kag_ge.expect_column_to_exist('Salary')


# In[69]:


kag_ge.expect_column_mean_to_be_between(
   'Salary', min_value=10_000, max_value=100_000)


# In[70]:


kag_ge.expect_column_values_to_be_between(
   'Salary', min_value=0, max_value=500_000)


# In[71]:


kag_ge.expect_column_values_to_not_be_null('Salary')


# In[72]:


kag_ge.expect_column_values_to_match_regex(
    'Country', r'America|India|Another|China')


# In[73]:


kag_ge.expect_column_values_to_be_of_type(
   'Salary', type_='int')


# In[74]:


kag_ge.save_expectation_suite('kaggle_expectations.json')


# In[75]:


kag_ge.to_csv('kag.csv')
import json
ge.validate(ge.read_csv('kag.csv'), 
    expectation_suite=json.load(
        open('kaggle_expectations.json')))


# ### How it works...

# ## Using pytest with pandas

# ### How to do it...

# ### How it works...

# ### There's more...

# ## Generating Tests with Hypothesis

# ### How to do it...

# ### How it works...
