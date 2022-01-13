# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 08:15:00 2021

@author: tosea
"""

import pandas as pd
from gensim.models import Word2Vec
from my_functions_NYT import *
import pickle
import nltk
from nltk import word_tokenize
from nltk.text import Text
the_path = "C:/Users/tosea/NYT_API/"

'''
df = pd.read_csv(r"NYT_text_1851_1980.csv", encoding="ISO-8859-1")
#print(df.head(15))
df["clean"] = [clean_text(i) for i in df.text]

df["clean_sw"] = df.clean.apply(rem_sw)

df["clean_sw_dict"] = df.clean_sw.apply(check_dictionary)

df["clean_sw_dict_stem"] = df.clean_sw_dict.apply(stem_fun)

df = give_year_labels(df)

df["clean_sw_dict_stem_adsw"] = df.clean_sw_dict_stem.apply(check_additional_sw)

df["num_tokens"] = df["clean_sw_dict_stem_adsw"].apply(lambda x: len(x.split(" ")))

pickle.dump(df, open("my_pd_dict_sw.pkl", "wb"))
'''


my_pd = pickle.load(open(f"{the_path}my_pd_dict_sw.pkl", "rb"))

'''
des_table = pd.concat([my_pd.groupby("label").count()["clean_sw_dict_stem_adsw"],
           my_pd.groupby("label").sum()["num_tokens"]], axis = 1)

des_table.to_csv("des_nyt_table.csv")


total_token_1851_1900 = get_token_in_years(my_pd, "clean_sw_dict_stem_adsw", "1851-1900")
total_token_1901_1930 = get_token_in_years(my_pd, "clean_sw_dict_stem_adsw", "1901-1930")
total_token_1931_1950 = get_token_in_years(my_pd, "clean_sw_dict_stem_adsw", "1931-1950")
total_token_1951_1960 = get_token_in_years(my_pd, "clean_sw_dict_stem_adsw", "1951-1960")
total_token_1961_1980 = get_token_in_years(my_pd, "clean_sw_dict_stem_adsw", "1961-1980")

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


'''


#df["jaccard_similarity"] = df.sw_stem.apply(
    #lambda x: jaccard_fun(x, "terroris"))




#train the model of sub
def top_similar_words(df, start_year, end_year, keyword, the_path, sg = 0):
    the_path = "C:/Users/tosea/NYT_API/"
    import pandas as pd
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    stm_keyword = stemmer.stem(keyword)
    my_pd_sub = df[ (start_year <= my_pd.year) & (my_pd.year <= end_year)]
    stem_model = extract_embeddings_domain(
        my_pd_sub.clean_sw_dict_stem_adsw, 300, the_path, f"clean_sw_dict_stem_{start_year}_{end_year}", sg)
    pickle.dump(stem_model, open(the_path + f"clean_sw_dict_stem_{start_year}_{end_year}_embeddings_domain_model{sg}.pkl", "wb"))
    dict_model = extract_embeddings_domain(
        my_pd_sub.clean_sw_dict, 300, the_path, f"clean_sw_dict_{start_year}_{end_year}", sg)
    pickle.dump(dict_model, open(the_path + f"clean_sw_dict_{start_year}_{end_year}_embeddings_domain_model{sg}.pkl", "wb"))
    stem_top_words_values = pd.DataFrame(
        [i for i in stem_model.wv.most_similar(stm_keyword, topn=150)],
        columns =[f'{stm_keyword}_stm', f'{stm_keyword}_Values_stm'])
    dict_top_words_values = pd.DataFrame(
        [i for i in dict_model.wv.most_similar(keyword, topn=150)],
        columns =[f'{keyword}_dict', f'{keyword}_Values_dict'])
    output = pd.concat([stem_top_words_values, dict_top_words_values], axis=1)
    return(output)



#result_1851_1900_terrorism = top_similar_words(my_pd, 1851, 1900, "terrorism", the_path)
#result_1901_1930_terrorism = top_similar_words(my_pd, 1901, 1930, "terrorism", the_path)
#result_1931_1950_terrorism = top_similar_words(my_pd, 1931, 1950, "terrorism", the_path)
#result_1951_1960_terrorism = top_similar_words(my_pd, 1951, 1960, "terrorism", the_path)
#result_1961_1980_terrorism = top_similar_words(my_pd, 1961, 1980, "terrorism", the_path)

#result_1851_1900_terrorist = top_similar_words(my_pd, 1851, 1900, "terrorist", the_path)
#result_1901_1930_terrorist = top_similar_words(my_pd, 1901, 1930, "terrorist", the_path)
#result_1931_1950_terrorist = top_similar_words(my_pd, 1931, 1950, "terrorist", the_path)
#result_1951_1960_terrorist = top_similar_words(my_pd, 1951, 1960, "terrorist", the_path)
#result_1961_1980_terrorist = top_similar_words(my_pd, 1961, 1980, "terrorist", the_path)

result_1851_1900_terrorist = top_similar_words(my_pd, 1851, 1900, "terrorist", the_path, 1)
result_1901_1930_terrorist = top_similar_words(my_pd, 1901, 1930, "terrorist", the_path, 1)
result_1931_1950_terrorist = top_similar_words(my_pd, 1931, 1950, "terrorist", the_path, 1)
result_1951_1960_terrorist = top_similar_words(my_pd, 1951, 1960, "terrorist", the_path, 1)
result_1961_1980_terrorist = top_similar_words(my_pd, 1961, 1980, "terrorist", the_path, 1)
'''
result_1851_1900_klan = top_similar_words(my_pd, 1851, 1900, "klan", the_path)
result_1901_1930_klan = top_similar_words(my_pd, 1901, 1930, "klan", the_path)
result_1931_1950_klan = top_similar_words(my_pd, 1931, 1950, "klan", the_path)
result_1951_1960_klan = top_similar_words(my_pd, 1951, 1960, "klan", the_path)
result_1961_1980_klan = top_similar_words(my_pd, 1961, 1980, "klan", the_path)

result_1851_1900_gue = top_similar_words(my_pd, 1851, 1900, "guerrilla", the_path)
result_1901_1930_gue = top_similar_words(my_pd, 1901, 1930, "guerrilla", the_path)
result_1931_1950_gue = top_similar_words(my_pd, 1931, 1950, "guerrilla", the_path)
result_1951_1960_gue = top_similar_words(my_pd, 1951, 1960, "guerrilla", the_path)
result_1961_1980_gue = top_similar_words(my_pd, 1961, 1980, "guerrilla", the_path)

result_1851_1900_anarchist = top_similar_words(my_pd, 1851, 1900, "anarchist", the_path)
result_1901_1930_anarchist = top_similar_words(my_pd, 1901, 1930, "anarchist", the_path)
result_1931_1950_anarchist = top_similar_words(my_pd, 1931, 1950, "anarchist", the_path)
result_1951_1960_anarchist = top_similar_words(my_pd, 1951, 1960, "anarchist", the_path)
result_1961_1980_anarchist = top_similar_words(my_pd, 1961, 1980, "anarchist", the_path)
'''

p1words = [i for i in result_1851_1900_terrorist.terrorist_stm]
p2words = [i for i in result_1901_1930_terrorist.terrorist_stm]
p3words = [i for i in result_1931_1950_terrorist.terrorist_stm]
p4words = [i for i in result_1951_1960_terrorist.terrorist_stm]
p5words = [i for i in result_1961_1980_terrorist.terrorist_stm]

print(p1words)
print(p2words)
print(p3words)
print(p4words)
print(p5words)

toatl_similar_words_calculated = p1words + p2words + p3words + p4words + p5words
#count total similar words frequencies
import collections
# using Counter to find frequency of elements
frequency = collections.Counter(toatl_similar_words_calculated)
print(dict(sorted(frequency.items(), key=lambda item: item[1],reverse=True)))


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



#my_vec_data = my_vec_fun(my_pd.clean_sw_dict_stem_adsw, 1, 1, the_path)
#my_tf_idf_data = my_tf_idf_fun(my_pd.clean_sw_dict_stem_adsw, 1, 1, the_path)
#my_pca_vec_i = my_pca_fun(my_tf_idf_data, 0.95, the_path)
#my_vec_data = pickle.load(open("my_vec_data.pkl", "rb"))
#model, metrics, my_preds = my_model_fun_grid(
#    my_vec_data, my_pd.label, the_path)


## Random Forest Classifier -- Keep All the Articles at the same amount
## We will keep the first period of all 885 articles, and randomly select 
## 885 articles from the rest of periods, respectively.
first_metirc = pd.DataFrame()
for i in range(3):

    sample1851 = my_pd[my_pd.label == "1851-1900"]
    sample1901 = my_pd[my_pd.label == "1901-1930"].sample(n = 885, replace=True)
    sample1931 = my_pd[my_pd.label == "1931-1950"].sample(n = 885, replace=True)
    sample1951 = my_pd[my_pd.label == "1951-1960"].sample(n = 885, replace=True)
    sample1961 = my_pd[my_pd.label == "1961-1980"].sample(n = 885, replace=True)
    
    sampled_pd = pd.concat([sample1851, sample1901, sample1931, 
                            sample1951, sample1961])
    
    sample_path = "C:/Users/tosea/NYT_API/sample/"
    my_vec_data = my_vec_fun(sampled_pd.clean_sw_dict_stem_adsw, 1, 1,sample_path)
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
