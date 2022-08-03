#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 5/25/2022

@author: Yu-Hsuan Liu
"""
import pandas as pd
from my_functions_NYT import *
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
from kneed import KneeLocator
import re
import random



the_path = "C:/Users/tosea/NYT_API/"

df = pd.read_csv("full_nyt_text.csv")
###check repeated titles ####
# the_title = "Terrorist Abductors of Moro Say They Are Carrying Out Death Plan"
# for i, j  in zip(df[df.title.str.contains(the_title)].index,
#                  df[df.title.str.contains(the_title)].text):
#     print(i, "#################\n")
                             
# df.sort_values("title").title.to_csv("titles.csv")
###check repeated titles ####


df_1851_1900 = df[ (1851 <= df.year) & (df.year <= 1900) ]
df_1901_1930 = df[ (1901 <= df.year) & (df.year <= 1930) ]
df_1931_1950 = df[ (1931 <= df.year) & (df.year <= 1950) ]
df_1951_1960 = df[ (1951 <= df.year) & (df.year <= 1960) ]
df_1961_1980 = df[ (1961 <= df.year) & (df.year <= 1980) ]



'''
df_1851_1900["clean_sw_adsw_bi"] = fetch_bi_grams(df_1851_1900.clean_sw_adsw.str.split())
df_1901_1930["clean_sw_adsw_bi"] = fetch_bi_grams(df_1901_1930.clean_sw_adsw.str.split())
df_1931_1950["clean_sw_adsw_bi"] = fetch_bi_grams(df_1931_1950.clean_sw_adsw.str.split())
df_1951_1960["clean_sw_adsw_bi"] = fetch_bi_grams(df_1951_1960.clean_sw_adsw.str.split())
df_1961_1980["clean_sw_adsw_bi"] = fetch_bi_grams(df_1961_1980.clean_sw_adsw.str.split())

df_1851_1900["clean_dict_sw_adsw_bi"] = fetch_bi_grams(df_1851_1900.clean_dict_sw_adsw.str.split())
df_1901_1930["clean_dict_sw_adsw_bi"] = fetch_bi_grams(df_1901_1930.clean_dict_sw_adsw.str.split())
df_1931_1950["clean_dict_sw_adsw_bi"] = fetch_bi_grams(df_1931_1950.clean_dict_sw_adsw.str.split())
df_1951_1960["clean_dict_sw_adsw_bi"] = fetch_bi_grams(df_1951_1960.clean_dict_sw_adsw.str.split())
df_1961_1980["clean_dict_sw_adsw_bi"] = fetch_bi_grams(df_1961_1980.clean_dict_sw_adsw.str.split())
'''



df_list = [df_1851_1900, df_1901_1930, df_1931_1950, df_1951_1960, df_1961_1980]
year_group = ["1851_1900", "1901_1930", "1931_1950", "1951_1960", "1961_1980"]



        
def coherence_score(data, colname):
    the_data = data[colname].str.split()
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    
    corpus = [id2word.doc2bow(text) for text in the_data]
    c_scores = list()
    for word in range(1, 10):
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=word, id2word=id2word, iterations=10, passes=5,
            random_state=123)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                              dictionary=dictionary,
                                              coherence='c_v')
        c_scores.append(coherence_model_lda.get_coherence())
    
    x = range(1, 10)
    kn = KneeLocator(x, c_scores,
                     curve='concave', direction='increasing')
    opt_topics = kn.knee

    print ("Optimal topics is", opt_topics)
    plt.plot(x, c_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return opt_topics


def topic_distribution(data, k, the_path, colname):
    model_name = f'{data=}'.split('=')[0]
    the_data=data[colname].str.split()
    id2word = corpora.Dictionary(the_data)
    corpus = [id2word.doc2bow(text) for text in the_data]
    ldamodel = pickle.load(open(the_path + model_name + "_lda_gensim.pkl", "rb"))
    output=list(ldamodel[corpus])[0]
    return(output)

def topic_modeling(data, k, the_path, colname, year):
    model_name = year
    the_data=data[colname].str.split()
    id2word = corpora.Dictionary(the_data)
    corpus = [id2word.doc2bow(text) for text in the_data]
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=k, id2word=id2word, iterations=50, passes=15,
        random_state=123, minimum_probability=0.0)
    pickle.dump(ldamodel, open(the_path + model_name+ "_lda_gensim.pkl", "wb"))
    topics = ldamodel.print_topics(num_words=10)
    for topic in topics:
        print(topic)
    topic_dis = list(ldamodel[corpus])[0]
    print("topic disctribution: ", topic_dis)
    return topics, topic_dis


def topic_vis(data, k, the_path, colname):
    import pyLDAvis.gensim_models
    import pyLDAvis
    model_name = f'{data=}'.split('=')[0]
    the_data=data[colname].str.split()
    dictionary = Dictionary(the_data)
    id2word = corpora.Dictionary(the_data)
    corpus = [id2word.doc2bow(text) for text in the_data]
    ldamodel = pickle.load(open(the_path + model_name + "_lda_gensim.pkl", "rb"))
    # visualization

def get_topics(df_list, v_column, the_path, year_group):
    co_score = []
    for i in df_list:
        co_score.append(coherence_score(i, v_column))
    print("coherence score: ", co_score)
    topic_list = []
    topic_distri = []
    for j,k,l in zip(df_list, co_score, year_group):
        a, b = topic_modeling(j, k, the_path, v_column, l)
        topic_list.append(a)
        topic_distri.append(b)
    return co_score, topic_list, topic_distri

'''
#bi
for i in df_list:
    coherence_score(i, "clean_sw_dict_stem_adsw_bi")

print(topic_modeling(df_1851_1900,1, the_path, "clean_sw_dict_stem_adsw_bi")) #1
print(topic_modeling(df_1901_1930,3, the_path, "clean_sw_dict_stem_adsw_bi")) #3
print(topic_modeling(df_1931_1950,3, the_path, "clean_sw_dict_stem_adsw_bi")) #3
print(topic_modeling(df_1951_1960,3, the_path, "clean_sw_dict_stem_adsw_bi")) #3
print(topic_modeling(df_1961_1980,4, the_path, "clean_sw_dict_stem_adsw_bi")) #4


#bi without removing dictinory and stemming
for i in df_list:
    coherence_score(i, "clean_sw_bi")

print(topic_modeling(df_1851_1900,2, the_path, "clean_sw_bi")) #2
print(topic_modeling(df_1901_1930,1, the_path, "clean_sw_bi")) #1
print(topic_modeling(df_1931_1950,1, the_path, "clean_sw_bi")) #1
print(topic_modeling(df_1951_1960,1, the_path, "clean_sw_bi")) #1
print(topic_modeling(df_1961_1980,4, the_path, "clean_sw_bi")) #4


for i in df_list:
    coherence_score(i, "clean_sw_adsw_bi")

print(topic_modeling(df_1851_1900,2, the_path, "clean_sw_adsw_bi")) #2
print(topic_modeling(df_1901_1930,1, the_path, "clean_sw_adsw_bi")) #1
print(topic_modeling(df_1931_1950,1, the_path, "clean_sw_adsw_bi")) #1
print(topic_modeling(df_1951_1960,1, the_path, "clean_sw_adsw_bi")) #1
print(topic_modeling(df_1961_1980,4, the_path, "clean_sw_adsw_bi")) #4

'''

#bi_gram topics
co_score, topic_list, topic_distribution = get_topics(
    df_list, "clean_dict_sw_adsw_stem_bi", the_path, year_group)

#tri_gram_topics
co_score_tri, topic_list_tri, topic_distribution_tri = get_topics(
    df_list, "clean_dict_sw_adsw_stem_tri", the_path, year_group)



import re
topic_words_list = []
for i in topic_list_tri:
    #print(i)
    for j in i:
        r1 = re.findall('\"\w*\"',j[1])
        topic_words_list.append(",".join(r1))

distribution_list = []
for k,l in zip(topic_distribution_tri, co_score_tri):
    
    for m in range(l):
        distribution_list.append(k[m][1])


year_column = []
k_column = []
for n,o in zip(year_group, co_score_tri):
    for p in range(o):
        year_column.append(n)
        k_column.append(o)
        

topic_table = pd.DataFrame(list(zip(year_column, k_column, topic_words_list, distribution_list)),
               columns =['Historical Periods', 
                         'k (optimized number of topics)', 
                         'Topic Words', 
                         'Topic Distribution'])

topic_table.to_csv("topic_table.csv")

#reorder wordlist by 1. years 2. topic distribution
topic_table = topic_table.sort_values(by = ["Historical Periods", "Topic Distribution"], ascending=[True, False])
my_topic_word_list = [re.sub("\"","", re.sub("\",\"",", ",i)) for i in topic_table["Topic Words"]]




#check the stemming words
# =============================================================================
# topic_list
# from nltk.stem.porter import PorterStemmer
# stemmer = PorterStemmer()
# stemmer.stem("Johnston")
# =============================================================================

# =============================================================================
# #check the topics and related articles
# =============================================================================


# (polit, vote, power, south, elect, present, parti, war, order, give)
# topic disctribution:  [(0, 1.0)]

def print_topic_texts(df):
    [print("###########HEAD############",
           "\nTITLE = ", title,
           "\nLINK = ", link,
           "\nPUB_DATE = ", pub_date,
           "\nIndex = ", ind,
           "\nQUOTE = \n",
           re.findall(r"([^.]*?terrorist[^.]*\.)", i) ,
           "\nTri Text = \n", tri_text,
           "***********END*************\n") for i, tri_text, title, link, pub_date, ind in zip(
               df.text, df.tri, df.title, df.link, df["date"], df.index)]
               
df.columns
# "polit, vote, power, south, elect, present, parti, war, order, give "
south_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("south")) &
    (df.clean_dict_sw_adsw_stem.str.contains("elect|polit|vote|war"))) &
    (df.year <= 1900)].sample(n=25, random_state=123)

print_topic_texts(south_filter)
                                                          
# (russia, terrorist, war, german, russian, american, polit, organ, present, nation)
# topic disctribution:  [(0, 1.0)]

russia_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("german")) &
    (df.clean_dict_sw_adsw_stem.str.contains("russia|american|war"))) &
    ((df.year >= 1901)&(df.year <= 1930))].sample(n=25, random_state=123)

print_topic_texts(russia_filter)
                 
# 'terrorist, organ, govern, union, trial, labor, employ, law, parti, public.',
union_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("union")) &
    (df.clean_dict_sw_adsw_stem.str.contains("labor|employ|america|organ"))) &
    ((df.year >= 1931)&(df.year <= 1950))].sample(n=25, random_state=123)

print_topic_texts(union_filter)


 # 'japanes, chines, french, german, war, american, japan, terrorist, militari, govern.',
 
jp_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("japanes")) &
    (df.clean_dict_sw_adsw_stem.str.contains("chines|french|german"))) &
    ((df.year >= 1931)&(df.year <= 1950))].sample(n=25, random_state=123)

print_topic_texts(jp_filter)
          


          
           
 # 'war, world, polit, american, nation, peac, parti, govern, present, power',
 
american_war_filter = df[((df.clean_dict_sw_adsw_stem.str.contains("american")) &
     (df.clean_dict_sw_adsw_stem.str.contains("war")) &
     (df.clean_dict_sw_adsw_stem.str.contains("govern|world|power|polit|peac|nation") &
      (~df.clean_dict_sw_adsw_stem.str.contains("jerusalem|britain|arab|jewish|british|polish|yugoslav|french|italian|german|greek|japanes|union|chines|employ")))) &
     ((df.year >= 1931)&(df.year <= 1950))].sample(n=25, random_state=123)

print_topic_texts(american_war_filter)
 
 
 # 'german, terrorist, pari, polish, parti, italian, french, hungarian, yugoslav, greek',
german_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("german")) &
    (df.clean_dict_sw_adsw_stem.str.contains("pari|polish|italian|french|hungarian|yugoslav|greek"))) &
    ((df.year >= 1931)&(df.year <= 1950))].sample(n=25, random_state=123)

print_topic_texts(german_filter)
 
 
# 'british, jewish, arab, terrorist, britain, jerusalem, polic, arm, militari, confer',
jew_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("jewish")) &
    (df.clean_dict_sw_adsw_stem.str.contains("arab|britain|british|herusalem"))) &
    ((df.year >= 1931)&(df.year <= 1950))].sample(n=25, random_state=123)

print_topic_texts(jew_filter)
                                

# "terrorist, french, british, polit, polic, militari, parti, nation, arm, war"    
french_b_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("french|british")) &
    (df.clean_dict_sw_adsw_stem.str.contains("polit|polic|militari|war|arm"))) &
    ((df.year >= 1951)&(df.year <= 1960))].sample(n=25, random_state=123)

print_topic_texts(french_b_filter)

test1 = 'polit, parti, nation, power, major, million, militari, econom, percent, govern'.split(", ")
random.sample(test1, 6)
test1.pop(1)
# 'polit, parti, nation, power, major, million, militari, econom, percent, govern'
# 'american, terrorist, world, intern, polit, public, peac, nation, polici, respons'
# 'live, come, life, think, book, young, american, school, world, famili'
# 'terrorist, polic, group, former, feder, case, polit, organ, investig, prison'
# 'south, black, militari, war, white, american, vietnames, fight, guerrilla, african'
# 'terrorist, british, polic, irish, fire, secur, area, night, wound, violenc'
# 'israel, arab, palestinian, terrorist, egypt, ran, jewish, jerusalem, jordan, peac'
                                   
# "polit, parti, nation, power, major, million, militari, econom, percent, govern"

#(2, '0.014*"polit" + 0.008*"militari" + 0.007*"american" + 0.007*"terrorist" + 
#0.006*"south" + 0.005*"nation" + 0.005*"parti" + 0.005*"support" + 0.005*"power" + 0.005*"major"')
#(3, '0.004*"american" + 0.004*"public" + 0.004*"world" + 0.003*"nation" + 
#0.003*"feder" + 0.003*"million" + 0.003*"think" + 0.003*"come" + 0.003*"terrorist" + 0.002*"becom"')

polit_nation_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("south")) &
     (df.clean_dict_sw_adsw_stem.str.contains("polit")) &
    (df.clean_dict_sw_adsw_stem.str.contains("american|power|nation|parti|militari"))) &
    ((df.year >= 1961)&(df.year <= 1980))].sample(n=25, random_state=123)

print_topic_texts(polit_nation_filter)

# "terrorist, israel, polic, group, arab, militari, british, polit, palestinian, secur"     

israel_filter = df[
    ((df.clean_dict_sw_adsw_stem.str.contains("")) &
    (df.clean_dict_sw_adsw_stem.str.contains(""))) &
    ((df.year >= 1961)&(df.year <= 1980))].sample(n=25, random_state=123)

print_topic_texts(israel_filter)
  



klan_filter = df[(df.clean_dict_sw_adsw_stem.str.contains("klansman"))]
print_topic_texts(klan_filter)


