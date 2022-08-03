# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 08:15:00 2021

@author: tosea

#conda activate spyder-cf
"""

import pandas as pd
from gensim.models import Word2Vec
from my_functions_NYT import *
import pickle
import nltk
from nltk import word_tokenize
from nltk.text import Text
import matplotlib.pyplot as plt
the_path = "C:/Users/tosea/NYT_API/"

'''
df = pd.read_csv(r"NYT_text_1851_1980.csv", encoding="ISO-8859-1")

df["clean"] = [clean_text(i) for i in df.text]

df["clean_dict"] = df.clean.apply(check_dictionary)

df["clean_dict_sw"] = df.clean_dict.apply(rem_sw)

df["clean_dict_sw_adsw"] = df.clean_dict_sw.apply(check_additional_sw)

df["clean_dict_sw_adsw_stem"] = df.clean_dict_sw_adsw.apply(stem_fun)

df["num_tokens"] = df.clean_dict_sw_adsw_stem.apply(num_tokens)

df = give_year_labels(df)

pickle.dump(df, open("my_pd_dict_sw_0730_klansman.pkl", "wb"))
'''

#Excluding non-related texts or articles, such as World News Summary, 
#TV shows, Books, Sports, Musicals, Films, ...etc
not_contain = ["The Screen:","Book Review","The Kong and I","TV:","TV ",
               "THE SCREEN","Screen","STAGE VIEW","Publishing:", "The World",
               "Public and Private Games","Paperback","ON TELEVISION",
               "News of the Theater","IN AND OUT OF BOOKS","Editors' Choice",
               "Books: ","Books--Authors","Books of","Books and Authors",
               "Books -- Authors","BOOK NOTES","BOOK ENDS","Best Sellers",
               "BEHIND THE BEST SELLERS","At the Movies",
               "Arts and Leisure Guide","About Real Estate","Advertising",
               "Answers to Weekly Quiz","Answers to Quiz","A Listing of","TV",
               "BOOKS OF THE TIMES","THE PLAY","MUSIC VIEW", "Books of The",
               "GOING OUT Guide", "Movies", "Paper backs",
               "Paperback Best Sellers", "Paperbacks", "World News Summarized",
               "Westchester/This Week","Long Island/ This Week", "Major News",
               "Television This Week","New Jersey/This Week", "Weekly News Quiz",
               "The Screen;", "What's in a Book Name?", "When Is a Movie So Bad",
               "Sports Editor's Mailbox: The Animals at Shea/Scoreboard at Yankee Stadium",
               "Sports News Briefs","Sports World Special","Sports of The Times", 
               "Sports of the Times","Summary", "Briefs", "World News BriEfs",
               "Stage: ", "A preview OF FALL BOOKS; ", "Books Authors; ", 
               "NEW PUBLICATONS", "NEW PUBLICATIONS", "New Books","OF MANY THINGS: ",
               "PREVIEW of FALL BOOKS", "Movie Mailbag", "NEW BOOKS",
               "Passionate Story of a Bandit:'Augusto Matraga' Is at 5th Avenue Cinema Movie From Brazil by Santos Arrives",
               "Action Movie Set in Latin America", "M-G-M FILLS ROLES IN TWO NEW FILMS;",
               "His Latest Volume of Collected Plays", "Musical Cartoon to Play At the Brooklyn Academy",
               "STUDENTS TO GIVE PLAYS; 4 Municipal Colleges to Present Drama Festival May 12",
               "The Theater: A Musical","'Minnie' on Music -- and Rain",
               "Amid Trials of Croatian Nationalists, a Satirical Musical Comedy in Zagreb Evokes Laughter Through Tears",
               "THE STADIUM SEASON; Management Overcomes Many Difficulties -- Need of Music in Wartime",
               "Drama: Adapting Wiesel's", "HITCHCOCK: MASTER MELODRAMATIST", "Terrorism Is Drama",
               "They Weren't Riotous Comedies", "Sports In America",
               "'Mary Burns, Fugitive,' the Melodrama of a Girl Who Loved a Murderer, at the Paramount.",
               "'Thunderbolt,' Entebbe Raid In Israeli Film","Other World Events","The Region",
               "3 1/2-Hour Film Based on Uris' Novel Opens", "Screen: ",
               "6 Film Studio's Vie Over Entebbe Raid", "STAGE VIEW",
               "CHARGES TERRORISM IN FILM INDUSTRY; Head of New Jersey",
               "FILM CONTROVERSY", "FILM VIEW", "Film Festival: ", "Film Fete:",
               "Film:", "Kennedy Center Drops Disputed Film", "SHOP TALK",
               "Shouldn't Suspense Films Do More Than Just Kill Time", "STAMPS",
               "'Viva Italia!,' Starring Vittorio Gassman, Returns to Day of Separate-Sketch Film:The Cast"]




the_path = "C:/Users/tosea/NYT_API/"

my_pd = pickle.load(open(f"{the_path}my_pd_dict_sw_0730_klansman.pkl", "rb"))

#excluding the only sigle word in title as "film" (index == 15424) and 
#"television" (index == 11867, 11874, 13516, 14336, 18744, 19544, 19603, 20110,
#20346, 20670, 21318, 21508, 21610, 21649, 21659, 21927, 22122, 22216)
excluding_index_list = [15424, 11867, 11874, 13516, 14336, 18744, 19544, 19603, 20110,
                        20346, 20670, 21318, 21508, 21610, 21649, 21659, 21927, 22122, 22216]

my_pd = my_pd.drop(excluding_index_list)
# exclude topics not related to discussion of terrorist events
# The text/title contains "terrorist" or "terrorism" 
my_pd = my_pd[(my_pd.text.str.contains("terrorist")|
         my_pd.text.str.contains("terrorism")|
         my_pd.title.str.contains("terrorist")|
         my_pd.title.str.contains("terrorism"))]


#exclude the titles regarding new books/TV list/Films/Movies/
#Mucial/Drama/Play/nerghborhood events/Weekly Quiz
my_pd = my_pd[~(my_pd.title.str.contains('|'.join(not_contain)))]



# Check the difference of using text or  clean_dict_sw_adsw_stem to subset the duplicated ones.
# There are four cases difference, 9692, 13830, 18763, and 19301. The conclusion is that
# We should use "clean_dict_sw_adsw_stem" to exclude the duplices because there are some 
# duplicated articles only can be identified after stemming. 
 
# dup = my_pd[my_pd.duplicated(subset=['text'])]
# dup2 = my_pd[my_pd.duplicated(subset=['clean_dict_sw_adsw_stem'])]
# dup3 = pd.concat([dup,dup2]).drop_duplicates(keep=False)

#drop same values, keep the first encountered row
my_pd = my_pd.drop_duplicates(subset=['clean_dict_sw_adsw_stem'], keep='first')

#drop nan values
my_pd = my_pd.dropna(subset=['clean_dict_sw_adsw_stem'])


def join_fun(var):
    return " ".join(var)

my_pd["clean_dict_sw_adsw_stem_bi"], my_pd["clean_dict_sw_adsw_stem_tri"] = fetch_bi_grams(
    my_pd.clean_dict_sw_adsw_stem.str.split())
my_pd["clean_dict_sw_adsw_stem_bi"] = my_pd.clean_dict_sw_adsw_stem_bi.apply(join_fun)
my_pd["clean_dict_sw_adsw_stem_tri"] = my_pd.clean_dict_sw_adsw_stem_tri.apply(join_fun)


my_pd.iloc[:, 3:].to_csv("full_nyt_text.csv")

############################################
column_used = "clean_dict_sw_adsw_stem_tri"
############################################


#descriptive tables
des_table = pd.concat([my_pd.groupby("label").count()[column_used],
           my_pd.groupby("label").sum()["num_tokens"]], axis = 1)

des_table.to_csv("des_nyt_table.csv")


total_token_1851_1900 = get_token_in_years(my_pd, column_used, "1851-1900")
total_token_1901_1930 = get_token_in_years(my_pd, column_used, "1901-1930")
total_token_1931_1950 = get_token_in_years(my_pd, column_used, "1931-1950")
total_token_1951_1960 = get_token_in_years(my_pd, column_used, "1951-1960")
total_token_1961_1980 = get_token_in_years(my_pd, column_used, "1961-1980")

probability_1851_1900 = token_to_probability_table(total_token_1851_1900)
probability_1901_1930 = token_to_probability_table(total_token_1901_1930)
probability_1931_1950 = token_to_probability_table(total_token_1931_1950)
probability_1951_1960 = token_to_probability_table(total_token_1951_1960)
probability_1961_1980 = token_to_probability_table(total_token_1961_1980)

probability_1851_1900.head(25).to_csv("probability_1851_1900.csv", index=False)
probability_1901_1930.head(25).to_csv("probability_1901_1930.csv", index=False)
probability_1931_1950.head(25).to_csv("probability_1931_1950.csv", index=False)
probability_1951_1960.head(25).to_csv("probability_1951_1960.csv", index=False)
probability_1961_1980.head(25).to_csv("probability_1961_1980.csv", index=False)


word_cloud_save(probability_1851_1900, "words_cloud_1851_1900", 50)
word_cloud_save(probability_1901_1930, "words_cloud_1901_1930", 50)
word_cloud_save(probability_1931_1950, "words_cloud_1931_1950", 50)
word_cloud_save(probability_1951_1960, "words_cloud_1951_1960", 50)
word_cloud_save(probability_1961_1980, "words_cloud_1961_1980", 50)



#df["jaccard_similarity"] = df.sw_stem.apply(
    #lambda x: jaccard_fun(x, "terrorist"))


#train the models of similar words to "terrorist"
#sg = 1: skip-gram model
#sg = 0: CBOW model


keyword = "terrorist"
#SKNG model similar words (bootstrapping for 1000 samples and get the averaging top 100)
result_1851_1900_terrorist = get_mean_from_boot(my_pd, column_used, 1851, 1900, keyword, the_path, 1)
result_1901_1930_terrorist = get_mean_from_boot(my_pd, column_used, 1901, 1930, keyword, the_path, 1)
result_1931_1950_terrorist = get_mean_from_boot(my_pd, column_used, 1931, 1950, keyword, the_path, 1)
result_1951_1960_terrorist = get_mean_from_boot(my_pd, column_used, 1951, 1960, keyword, the_path, 1)
result_1961_1980_terrorist = get_mean_from_boot(my_pd, column_used, 1961, 1980, keyword, the_path, 1)

#CBOW model similar words (bootstrapping for 1000 samples and get the averaging top 100)
result_1851_1900_terrorist_0 = get_mean_from_boot(my_pd, column_used, 1851, 1900, keyword, the_path, 0)
result_1901_1930_terrorist_0 = get_mean_from_boot(my_pd, column_used, 1901, 1930, keyword, the_path, 0)
result_1931_1950_terrorist_0 = get_mean_from_boot(my_pd, column_used, 1931, 1950, keyword, the_path, 0)
result_1951_1960_terrorist_0 = get_mean_from_boot(my_pd, column_used, 1951, 1960, keyword, the_path, 0)
result_1961_1980_terrorist_0 = get_mean_from_boot(my_pd, column_used, 1961, 1980, keyword, the_path, 0)

def print_klux(df):
    return df[df.terrorist_stm.str.contains("klan")]

print_klux(result_1851_1900_terrorist)
print_klux(result_1901_1930_terrorist)
print_klux(result_1931_1950_terrorist)
print_klux(result_1951_1960_terrorist)
print_klux(result_1961_1980_terrorist)

print_klux(result_1851_1900_terrorist_0)
print_klux(result_1901_1930_terrorist_0)
print_klux(result_1931_1950_terrorist_0)
print_klux(result_1951_1960_terrorist_0)
print_klux(result_1961_1980_terrorist_0)


top_head_similars = 100
#combine the similar words from the two models
simi_total = pd.concat([result_1851_1900_terrorist.head(top_head_similars),
                        result_1901_1930_terrorist.head(top_head_similars), 
                        result_1931_1950_terrorist.head(top_head_similars), 
                        result_1951_1960_terrorist.head(top_head_similars),
                        result_1961_1980_terrorist.head(top_head_similars),
                        result_1851_1900_terrorist_0.head(top_head_similars),
                        result_1901_1930_terrorist_0.head(top_head_similars),
                        result_1931_1950_terrorist_0.head(top_head_similars),
                        result_1951_1960_terrorist_0.head(top_head_similars), 
                        result_1961_1980_terrorist_0.head(top_head_similars)])

simi_total.to_csv(f"{the_path}simi_total.csv", index = False)


### SKNG similar words################################################





p1words = [i for i in result_1851_1900_terrorist.head(top_head_similars).terrorist_stm]
p2words = [i for i in result_1901_1930_terrorist.head(top_head_similars).terrorist_stm]
p3words = [i for i in result_1931_1950_terrorist.head(top_head_similars).terrorist_stm]
p4words = [i for i in result_1951_1960_terrorist.head(top_head_similars).terrorist_stm]
p5words = [i for i in result_1961_1980_terrorist.head(top_head_similars).terrorist_stm]


total_similar_words_list_1 = [" ".join(p1words),
                            " ".join(p2words),
                            " ".join(p3words),
                            " ".join(p4words),
                            " ".join(p5words)]

# #####Calculate Jaccard Similarity########
jac_table_1 = cal_jaccard_similarity(total_similar_words_list_1)
jac_table_1.to_csv("jac_table_1.csv", index=False)
####################

#####Calculate the similarity between similar words from different periods#####
tf_idf_table_1 = cal_tf_idf_cosine_similarity(total_similar_words_list_1)
tf_idf_table_1.to_csv("tf_idf_table_1.csv", index=False)
#############################################################################
result_1851_1900_terrorist.head(top_head_similars).to_csv("result_1851_1900_terrorist.csv", index=False)
result_1901_1930_terrorist.head(top_head_similars).to_csv("result_1901_1930_terrorist.csv", index=False)
result_1931_1950_terrorist.head(top_head_similars).to_csv("result_1931_1950_terrorist.csv", index=False)
result_1951_1960_terrorist.head(top_head_similars).to_csv("result_1951_1960_terrorist.csv", index=False)
result_1961_1980_terrorist.head(top_head_similars).to_csv("result_1961_1980_terrorist.csv", index=False)

##calculate the repeatly shown similar words across the five historical periods#####
toatl_similar_words_calculated_1 = p1words + p2words + p3words + p4words + p5words
#count total similar words frequencies
import collections
# using Counter to find frequency of elements
frequency_1 = collections.Counter(toatl_similar_words_calculated_1)
SKNG_similars = dict(sorted(frequency_1.items(), key=lambda item: item[1],reverse=True))
######################################################################
#result_1851_1900_terrorist_0


### CBOW similar words ################################################
p1words_0 = [i for i in result_1851_1900_terrorist_0.head(top_head_similars).terrorist_stm]
p2words_0 = [i for i in result_1901_1930_terrorist_0.head(top_head_similars).terrorist_stm]
p3words_0 = [i for i in result_1931_1950_terrorist_0.head(top_head_similars).terrorist_stm]
p4words_0 = [i for i in result_1951_1960_terrorist_0.head(top_head_similars).terrorist_stm]
p5words_0 = [i for i in result_1961_1980_terrorist_0.head(top_head_similars).terrorist_stm]

total_similar_words_list_0 = [" ".join(p1words_0),
                            " ".join(p2words_0),
                            " ".join(p3words_0),
                            " ".join(p4words_0),
                            " ".join(p5words_0)]

result_1851_1900_terrorist_0.head(top_head_similars).to_csv("result_1851_1900_terrorist_0.csv", index=False)
result_1901_1930_terrorist_0.head(top_head_similars).to_csv("result_1901_1930_terrorist_0.csv", index=False)
result_1931_1950_terrorist_0.head(top_head_similars).to_csv("result_1931_1950_terrorist_0.csv", index=False)
result_1951_1960_terrorist_0.head(top_head_similars).to_csv("result_1951_1960_terrorist_0.csv", index=False)
result_1961_1980_terrorist_0.head(top_head_similars).to_csv("result_1961_1980_terrorist_0.csv", index=False)

# #####Calculate Jaccard Similarity########
jac_table_0 = cal_jaccard_similarity(total_similar_words_list_0)
jac_table_0.to_csv("jac_table_0.csv", index=False)
####################

#####Calculate the similarity between similar words from different periods#####
tf_idf_table_0 = cal_tf_idf_cosine_similarity(total_similar_words_list_0)
tf_idf_table_0.to_csv("tf_idf_table_0.csv", index=False)


##calculate the repeatly shown similar words across the five historical periods#####
toatl_similar_words_calculated_0 = p1words_0 + p2words_0 + p3words_0 + p4words_0 + p5words_0
# using Counter to find frequency of elements
frequency_0 = collections.Counter(toatl_similar_words_calculated_0)
CBOW_similars = dict(sorted(frequency_0.items(), key=lambda item: item[1],reverse=True))

# =============================================================================
# print out the text description for figure 3 and 4 in the manuscrpit
# =============================================================================

d = {'1851-1900 SKNG': p1words, '1901_1930 SKNG': p2words,
     '1931_1950 SKNG': p3words, '1951_1960 SKNG': p4words,
     '1961_1980 SKNG': p5words,
     '1851-1900 CBOW': p1words_0, '1901_1930 CBOW': p2words_0,
     '1931_1950 CBOW': p3words_0, '1951_1960 CBOW': p4words_0,
     '1961_1980 CBOW': p5words_0}

total_similar_words_list = pd.DataFrame(data=d)
total_similar_words_list.to_csv("total_similar_words_list.csv", index = False)
#print the text in the writen document
for i in total_similar_words_list.columns:
    print(f"Between {i[0:4]} and {i[5:9]}, the top twenty-five most similar words to 'terrorist' in the {i[10:14]} model are:",
          ", ".join(total_similar_words_list.head(25)[i])+".", end=' ')

# =============================================================================


# =============================================================================
# print out the text description for figure 5 in the manuscrpit
# =============================================================================
import inflect

p = inflect.engine()

simi_words_count_over_3 = pd.read_csv("top_100_simi_words_counts_over_three.csv", index_col=False)["x"]

#", ".join(pd.read_csv("top_100_simi_words_counts_over_three.csv", index_col=False)["x"])

print(f"In Figure 5, there are {p.number_to_words(len(simi_words_count_over_3))} words in the graph. Those words are the top two-hundred similar words which are counted over four times across the five historical periods. These words represent the repeatedly shown elements of the word “terrorist” across the five historical periods (1851-1900, 1901-1930, 1931-1950, 1951-1960 and 1961-1980) in NYT news texts. Those words are: {', '.join(simi_words_count_over_3)}.")

'''

print([i for i in result_1851_1900_klan.klan_stm])
print([i for i in result_1901_1930_klan.klan_stm])
print([i for i in result_1931_1950_klan.klan_stm])
print([i for i in result_1951_1960_klan.klan_stm])
print([i for i in result_1961_1980_klan.klan_stm])

[print(i) for i in result_1851_1900_klan.klan_dict]
[print(i) for i in result_1901_1930_klan.klan_dict]
[print(i) for i in result_1931_1950_klan.klan_dict]
[print(i) for i in result_1951_1960_klan.klan_dict]
[print(i) for i in result_1961_1980_klan.klan_dict]

print([i for i in result_1851_1900_gue.guerrilla_stm])
print([i for i in result_1901_1930_gue.guerrilla_stm])
print([i for i in result_1931_1950_gue.guerrilla_stm])
print([i for i in result_1951_1960_gue.guerrilla_stm])
print([i for i in result_1961_1980_gue.guerrilla_stm])

print([i for i in result_1851_1900_anarchist.anarchist_stm])
print([i for i in result_1901_1930_anarchist.anarchist_stm])
print([i for i in result_1931_1950_anarchist.anarchist_stm])
print([i for i in result_1951_1960_anarchist.anarchist_stm])
print([i for i in result_1961_1980_anarchist.anarchist_stm])

print([i for i in result_1851_1900_anarchist.anarchist_dict])
print([i for i in result_1901_1930_anarchist.anarchist_dict])
print([i for i in result_1931_1950_anarchist.anarchist_dict])
print([i for i in result_1951_1960_anarchist.anarchist_dict])
print([i for i in result_1961_1980_anarchist.anarchist_dict])


import nltk
dictionary = nltk.corpus.words.words("en")
"ku klux klan"  in dictionary

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem("anarchist") 

print_numbers = 50


print("################################################")
print(result_1851_1900_terrorism.head(print_numbers))
print("__________________________________")
print(result_1901_1930_terrorism.head(print_numbers))
print("__________________________________")
print(result_1931_1950_terrorism.head(print_numbers))
print("__________________________________")
print(result_1951_1960_terrorism.head(print_numbers))
print("__________________________________")
print(result_1961_1980_terrorism.head(print_numbers))

print("################################################")

print("################################################")
print(result_1851_1900_terrorist.head(print_numbers))
print("__________________________________")
print(result_1901_1930_terrorist.head(print_numbers))
print("__________________________________")
print(result_1931_1950_terrorist.head(print_numbers))
print("__________________________________")
print(result_1951_1960_terrorist.head(print_numbers))
print("__________________________________")
print(result_1961_1980_terrorist.head(print_numbers))
print("################################################")

'''



#my_vec_data = my_vec_fun(my_pd.clean_dict_sw_adsw_stem, 1, 1, the_path)
#my_tf_idf_data = my_tf_idf_fun(my_pd.clean_dict_sw_adsw_stem, 1, 1, the_path)
#my_pca_vec_i = my_pca_fun(my_tf_idf_data, 0.95, the_path)
#my_vec_data = pickle.load(open("my_vec_data.pkl", "rb"))
#model, metrics, my_preds = my_model_fun_grid(
#    my_vec_data, my_pd.label, the_path)




# Google Model similarity


def my_vec_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    import pickle
    vectorizer = CountVectorizer(ngram_range=(m, n))
    my_vec_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_vec_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "vectorizer.pkl", "wb" ))
    pickle.dump(my_vec_t, open(path_o + "my_vec_data.pkl", "wb" ))
    return my_vec_t

def my_tf_idf_fun(df_in, m, n, path_o):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle
    vectorizer = TfidfVectorizer(ngram_range=(m, n))
    my_tf_idf_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_tf_idf_t.columns = vectorizer.vocabulary_
    pickle.dump(vectorizer, open(path_o + "tf_idf.pkl", "wb" ))
    pickle.dump(my_tf_idf_t, open(path_o + "my_tf_idf_data.pkl", "wb" ))
    return my_tf_idf_t

def my_model_fun(df_in, label_in, path_o):
    #function 1
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    import pickle
    my_model = RandomForestClassifier(max_depth=10, random_state=123)
    #my_model = GaussianNB()
    #80/20 train,test,split
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=123)
    my_model.fit(X_train, y_train)
    pickle.dump(my_model, open(path_o + "my_model.pkl", "wb"))
    y_pred = my_model.predict(X_test)
    model_metrics = pd.DataFrame(precision_recall_fscore_support(
    y_test, y_pred, average='weighted'))
    model_metrics.index = ["precision", "recall", "fscore", "none"]
    #function 2 prediction
    the_preds = pd.DataFrame(my_model.predict_proba(X_test))
    the_preds.columns = my_model.classes_

    return my_model, model_metrics, the_preds

def my_model_fun_grid(df_in, label_in, path_o):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    import pickle
    from sklearn.model_selection import GridSearchCV
    #my_model = RandomForestClassifier(max_depth=10, random_state=123)
    my_model = RandomForestClassifier(random_state=123)
    parameters = {'n_estimators':[10, 100], 'max_depth':[None, 1, 10],
                  'random_state': [123], 'criterion': ("gini", "entropy")}
    my_grid = GridSearchCV(my_model, parameters)
    #80/20 train,test,split
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=123)
    my_grid.fit(X_train, y_train)
    print ("Best Score:", my_grid.best_score_)
    best_params = my_grid.best_params_
    my_model_opt = RandomForestClassifier(**best_params)
    #my_model = GaussianNB()
    my_model_opt.fit(X_train, y_train)
    pickle.dump(my_model_opt, open(path_o + "my_model.pkl", "wb"))
    y_pred = my_model_opt.predict(X_test)
    model_metrics = pd.DataFrame(precision_recall_fscore_support(
    y_test, y_pred, average='weighted'))
    model_metrics.index = ["precision", "recall", "fscore", "none"]
    #function 2 prediction
    the_preds = pd.DataFrame(my_model_opt.predict_proba(X_test))
    the_preds.columns = my_model_opt.classes_

    return my_model_opt, model_metrics, the_preds

def my_pca_fun(df_in, n_comp_in, path_o):
    from sklearn.decomposition import PCA
    import pickle
    my_pca = PCA(n_components=n_comp_in)
    my_pca_vec = my_pca.fit_transform(df_in)
    pickle.dump(my_pca, open(path_o + "pca.pkl", "wb"))
    return my_pca_vec


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_tfidf_similarity():
	vectorizer = TfidfVectorizer()

	# To make uniformed vectors, both documents need to be combined first.
	documents.insert(0, base_document)
	embeddings = vectorizer.fit_transform(documents)

	cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    
## Random Forest Classifier -- Keep All the Articles at the same amount
## We will keep count numer the first period of all articles, and randomly select 
## that count numbers of articles from the rest of periods.
first_metirc = pd.DataFrame()

for i in range(1000):

    sample1851 = my_pd[my_pd.label == "1851-1900"]
    sample1901 = my_pd[my_pd.label == "1901-1930"].sample(n = sample1851.shape[0], replace=True)
    sample1931 = my_pd[my_pd.label == "1931-1950"].sample(n = sample1851.shape[0], replace=True)
    sample1951 = my_pd[my_pd.label == "1951-1960"].sample(n = sample1851.shape[0], replace=True)
    sample1961 = my_pd[my_pd.label == "1961-1980"].sample(n = sample1851.shape[0], replace=True)
    
    sampled_pd = pd.concat([sample1851, sample1901, sample1931, 
                            sample1951, sample1961])
    
    sample_path = "C:/Users/tosea/NYT_API/sample/"
    my_vec_data = my_vec_fun(sampled_pd[column_used], 1, 1,sample_path)
    model, metrics, my_preds = my_model_fun_grid(my_vec_data, sampled_pd.label, sample_path)

    first_metirc = pd.concat([first_metirc, metrics.T])

print(first_metirc.mean(axis = 0))
print(first_metirc.std(axis = 0))

#from https://www.fbi.gov/history/famous-cases/wall-street-bombing-1920
sample_text = "Within minutes, the cart exploded into a hail of metal fragments—immediately killing more than 30 people and injuring some 300. The carnage was horrific, and the death toll kept rising as the day wore on and more victims succumbed. Who was responsible? In the beginning it wasn’t obvious that the explosion was an intentional act of terrorism. Crews cleaned the damage up overnight, including physical evidence that today would be crucial to identifying the perpetrator. By the next morning Wall Street was back in business—broken windows draped in canvass, workers in bandages, but functioning none-the-less. Conspiracy theories abounded, but the New York Police and Fire Departments, the Bureau of Investigation (our predecessor), and the U.S. Secret Service were on the job. Each avidly pursued leads. The Bureau interviewed hundreds of people who had been around the area before, during, and after the attack, but developed little information of value. The few recollections of the driver and wagon were vague and virtually useless. The NYPD was able to reconstruct the bomb and its fuse mechanism, but there was much debate about the nature of the explosive, and all the potential components were commonly available. The most promising lead had actually come prior to the explosion. A letter carrier had found four crudely spelled and printed flyers in the area, from a group calling itself the “American Anarchist Fighters” that demanded the release of political prisoners. The letters, discovered later, seemed similar to ones used the previous year in two bombing campaigns fomented by Italian Anarchists. The Bureau worked diligently, investigating up and down the East Coast, to trace the printing of these flyers, without success. Based on bomb attacks over the previous decade, the Bureau initially suspected followers of the Italian Anarchist Luigi Galleani. But the case couldn’t be proved, and the anarchist had fled the country. Over the next three years, hot leads turned cold and promising trails turned into dead ends. In the end, the bombers were not identified. The best evidence and analysis since that fateful "
my_pred, my_pred_score = score_text(sample_text, sample_path)

#from https://www.digitalhistory.uh.edu/topic_display.cfm?tcid=94
sample_text2 = "Terrorist tactics were subsequently adopted by some dissident groups in the Ottoman and British empire and by some anarchists in the United States and Western Europe. Late nineteenth- and early-twentieth-century terrorism typically took the form of assassination attempts on heads of state and bomb attacks on public buildings. Between 1880, the president of France, a Spanish prime minister, an Austrian empress, an Italian king, and two U.S. presidents were assassinated. Attempts were also made on the life of a German chancellor and emperor."
my_pred2, my_pred_score2 = score_text(sample_text2, sample_path)

sample_text3 = "Saigon, Feb. 7-Vict Cong terrorism against cil'ilians, Viet1a,nese and American, J,as in: reased in recent weeks, particularly against local offichds ~nd national policemen. Res11011sible sources said lhe Communists are apparenlly trying to intimidate those who are working !or the  de1·elopment program which would remol'e villages and hamlets lrom Viel Cong influence. Confirmed counts of civiliam killed, wounded and abducted were considerably higher in ll11 week which ended February 4 than in the previous w~ck, according lo statistics releasec today. Bomb Rocks Party I .~ :?adJ.fJnn,.nrrsnn~ were killed in three incidents report-I ed today, including the bombing of a pre-Tct party last nigl1I. The blast occurred at the home of the Kontum province deputy chief for security. Three persons died and six were wounded. The official was not reported injured. Early yesterday live Viet Cong invaded the home of a school teacher in a coastal village north of Tam Ky and assassinated the teacher. I Amon'g the 3~ persons killed, reported n last week's compilation, were 2 national policemen, 2 hamlet officials, 4 hamlel chiefs, the son of one hamlet chie{ and the wi(c o{ another, and a Hoa Hao Buddhist leader."
my_pred3, my_pred_score3 = score_text(sample_text3, sample_path)


sample_text4 = " The most renrnrlmble Incident of this most remnrlmble strike is thnt lhe mnstei·s owed the men nothing. but three cln~·s' wages were giYen ns nu advance. the strike therefore being kept going with the money of the emplo~·ers, ugninst whom the strl\rn wns ostensibly directed. Not nil the mlllowners were willing to pny the str!kers. At Ilnron IIelntzel's mill 11 threatening crowd :issembled :ind demanded to Ree the proprietor. 'fhe officer in comrnnnd of the military ordered them to cllspcrsf.' within 10 mini1tes, otherwise he would use weapons. 'When the time expired the crowd refused to 1110\'e, null the oUiccr orllerc1l his men to lond their guns. Women rnshecl from the crowd noel threw back theh· shawls to e:q>ose their breasts lo the bullets. A small boy produced a woo1l <'ruclllx hearing the words, .. This ls the only weapon we ha\'e. Just ns the soldiers were about to fire Il:iron lleintzcl appeareu, and to prevent bloodshed  Terrorism also Is at work. :\I. Stelgert, n mlllowner employing UOO persons, refused to pa~• the strikers. Later be was ,·islted b~· four strnugers, who drew revolnirs ancl tbrearenecl to kill him If he dicl not gl\•~ his word of honor t!) pa~• the work people tomorrow morning. They warned him lf he applied for help to the mllitnrr he would be :t dead man Inside of four days. 'l'he men ere not Lodz workmen, but prolmbly emissaries of the revolutionury "
my_pred4, my_pred_score4 = score_text(sample_text4, sample_path)



'''
#If we want to cut N words before/after terrorist  (unfinished)

less_1899_pd = my_pd[(my_pd.year <= 1899) & (my_pd.num_tokens >= 2000)]

("terrorist") in less_1899_pd.clean_sw_dict[2]

first_position = less_1899_pd.clean_sw_dict[2].split().index("terrorist")

up_bound = first_position - 500

less_1899_pd.clean_sw_dict[2].partition("terrorism")[0]

first_position = less_1899_pd.clean_sw_dict[2].split().index("terrorism")

'''
'''
def get_word_range(keyword, text, n):
    no_terrorist = False
    no_terrorism = False
    both_terrorism_terrorist = False
    
    #try to get the first postion of the word to set the upper bound
    try:
        terrorist_position = text.split().index("terrorist")
    except ValueError:
        terrorist_position = 999999
        no_terrorist = True
    try:
        terrorism_position = text.split().index("terrorism")
    except ValueError:
        terrorism_position = 999999
        no_terrorism = True
    
    if (no_terrorist == False) & (no_terrorism == False):
        both_terrorism_terrorist == True
    
    if terrorism_position < terrorist_position:
        first_position = terrorism_position
        
    elif terrorist_position > terrorism_position:
        first_position = teterrorist_position
    #upper bound set
    up_bound = first_position - n
    
    last_position = first_position
    while ("terrorism" or "terrorist") in text:
        if both_terrorism_terrorist == True:
            text = text.partition(text.split()[first_position])[2]
        
        

#    lower_bound = last position + n
    cutted_text = text(upper_bound:lower_bound)
    return cutted_text
'''
