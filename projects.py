#!/usr/bin/env python
# coding: utf-8

# # Makine Öğrenmesi ile Film Öneri Sistemi

# * Öneri sistemleri, veri biliminin en popüler uygulamaları arasındadır. Bir kullanıcının bir öğeye vereceği puanı veya tercihi tahmin etmek için kullanılırlar.
# 
# * Neredeyse her büyük şirket bunları bir şekilde uygulamıştır: Amazon bunu müşterilere ürün önermek için, YouTube otomatik oynatmada hangi videonun daha sonra oynatılacağına karar vermek için ve Facebook beğenilecek sayfalar ve takip edilecek kişiler önermek için kullanır.

# ## Kendi öneri sistemimizi oluşturalım

# * Bu Veri Bilimi projemde, basit ve içerik tabanlı öneri sistemlerinin temel bir modelini nasıl oluşturacağınızı göstereceğim.
# 
# * Bu modeller karmaşıklık, kalite veya doğruluk açısından endüstri standardına yakın olmayacak olsa da, daha iyi sonuçlar üreten daha karmaşık modeller oluşturmaya başlamanıza yardımcı olacaktır.
# 
# * Bu film öneri modelini oluşturmak için ihtiyacınız olan veri setlerini buradan indirebilirsiniz:
# * https://www.kaggle.com/datasets/ademylz/tmdb-5000-movies
# * https://www.kaggle.com/datasets/ademylz/tmdb-5000-credits

# In[135]:


import pandas as pd
import numpy as np
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits.head()


# In[10]:


movies.head()


# In[137]:


print("Credits:",credits.shape)
print("Movies Dataframe:",movies.shape)


# In[139]:


credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on = "id")
print(movies_merge.head())


# In[141]:


movies_cleaned_df = movies_merge.drop(columns = ["homepage", "title_x", "title_y", "status","production_countries"])
print(movies_cleaned_df.head())
print(movies_cleaned_df.info())
print(movies_cleaned_df.head(1)["overview"])


# ### İçerik Tabanlı Öneri Sistemi

# Şimdi, genel bakış sütununda verilen filmin konu özetlerini temel alarak bir öneri yapalım. Yani kullanıcımız bize bir film adı verirse, amacımız benzer konu özetlerini paylaşan filmleri önermektir.

# In[143]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df = 3, max_features = None,
                      strip_accents = "unicode", analyzer = "word", token_pattern = r'\w{1,}',
                      ngram_range = (1, 3),
                      stop_words = "english")


# In[145]:


# TF-IDF'yi "genel bakış" metnine uydurma
tfv_matrix = tfv.fit_transform(movies_cleaned_df['overview'].values.astype('U'))
print(tfv_matrix)
print(tfv_matrix.shape)


# In[147]:


from sklearn.metrics.pairwise import sigmoid_kernel

# Sigmoid çekirdeği hesaplayalım
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
print(sig[0])


# ### Dizinlerin ve Film Başlıklarının Ters Eşlenmesi

# In[149]:


# Dizinlerin ve film başlıklarının ters eşlenmesi
indices = pd.Series(movies_cleaned.index, index = movies_cleaned["original_title"]).drop_duplicates()
print(indices.head())
print(indices["Newlyweds"])
print(sig[4799])
similarities = list(enumerate(sig[indices["Newlyweds"]]))

# Listenin tamamı yerine ilk 10 elemanı göster
sorted_similarities = sorted(similarities, key = lambda x: x[1], reverse = True)

# İlk 10 sonucu göster ve geri kalanını gizle
print(sorted_similarities[:10])


# In[151]:


def give_recomendations(title, sig = sig):
    # original_title'a karşılık gelen dizini alalım
    idx = indices[title]

    # Pairwsie benzerlik puanlarını alalım
    sig_scores = list(enumerate(sig[idx]))

    # Filmleri sıralayalım
    sig_scores = sorted(sig_scores, key = lambda x: x[1], reverse = True)

    # En çok benzeyen 10 filmin puanları
    sig_scores = sig_scores[1:11]

    # Film endeksleri
    movie_indices = [i[0] for i in sig_scores]

    # En çok benzeyen 10 film
    return movies_cleaned["original_title"].iloc[movie_indices]


# #### İçerik tabanlı öneri sistemimizi benim çok beğendiğim Dövüş Kulübü filmiyle test edelim

# In[153]:


print(give_recomendations("Fight Club"))

