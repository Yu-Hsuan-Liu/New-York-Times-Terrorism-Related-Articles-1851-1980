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

#df = pd.read_csv(r"NYT_text_1851_1980.csv", encoding="ISO-8859-1")
#print(df.head(15))
#df["clean"] = [clean_text(i) for i in df.text]

#df["clean_sw"] = df.clean.apply(rem_sw)

#df["clean_sw_dict"] = df.clean_sw.apply(check_dictionary)

#df["clean_sw_dict_stem"] = df.clean_sw_dict.apply(stem_fun)

#df = give_year_labels(df)

#df["clean_sw_dict_stem_adsw"] = df.clean_sw_dict_stem.apply(check_additional_sw)

#df["num_tokens"] = df["clean_sw_dict_stem_adsw"].apply(lambda x: len(x.split(" ")))

#pickle.dump(df, open("my_pd_dict_sw.pkl", "wb"))



my_pd = pickle.load(open("my_pd_dict_sw.pkl", "rb"))


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





#df["jaccard_similarity"] = df.sw_stem.apply(
    #lambda x: jaccard_fun(x, "terroris"))


the_path = "C:/Users/tosea/NYT_API/"



#train the model of sub
def top_similar_words(df, start_year, end_year, keyword, the_path):
    the_path = "C:/Users/tosea/NYT_API/"
    import pandas as pd
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    stm_keyword = stemmer.stem(keyword)
    my_pd_sub = df[ (start_year <= my_pd.year) & (my_pd.year <= end_year)]
    stem_model = extract_embeddings_domain(
        my_pd_sub.clean_sw_dict_stem, 300, the_path, f"clean_sw_dict_stem_{start_year}_{end_year}")
    pickle.dump(stem_model, open(the_path + f"clean_sw_dict_stem_{start_year}_{end_year}_embeddings_domain_model.pkl", "wb"))
    dict_model = extract_embeddings_domain(
        my_pd_sub.clean_sw_dict, 300, the_path, f"clean_sw_dict_{start_year}_{end_year}")
    pickle.dump(dict_model, open(the_path + f"clean_sw_dict_{start_year}_{end_year}_embeddings_domain_model.pkl", "wb"))
    stem_top_words_values = pd.DataFrame(
        [i for i in stem_model.wv.most_similar(stm_keyword, topn=100)],
        columns =[f'{stm_keyword}_stm', f'{stm_keyword}_Values_stm'])
    dict_top_words_values = pd.DataFrame(
        [i for i in dict_model.wv.most_similar(keyword, topn=100)],
        columns =[f'{keyword}_dict', f'{keyword}_Values_dict'])
    output = pd.concat([stem_top_words_values, dict_top_words_values], axis=1)
    return(output)



result_1851_1900_terrorism = top_similar_words(my_pd, 1851, 1900, "terrorism", the_path)
result_1901_1930_terrorism = top_similar_words(my_pd, 1901, 1930, "terrorism", the_path)
result_1931_1950_terrorism = top_similar_words(my_pd, 1931, 1950, "terrorism", the_path)
result_1951_1960_terrorism = top_similar_words(my_pd, 1951, 1960, "terrorism", the_path)
result_1961_1980_terrorism = top_similar_words(my_pd, 1961, 1980, "terrorism", the_path)

result_1851_1900_terrorist = top_similar_words(my_pd, 1851, 1900, "terrorist", the_path)
result_1901_1930_terrorist = top_similar_words(my_pd, 1901, 1930, "terrorist", the_path)
result_1931_1950_terrorist = top_similar_words(my_pd, 1931, 1950, "terrorist", the_path)
result_1951_1960_terrorist = top_similar_words(my_pd, 1951, 1960, "terrorist", the_path)
result_1961_1980_terrorist = top_similar_words(my_pd, 1961, 1980, "terrorist", the_path)

'''
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


sample_text = ""
my_pred, my_pred_score = score_text(sample_text, the_path)

# Google Model similarity
from gensim.models import KeyedVectors
the_path_in = "C:/Users/tosea/NLP@Columbia/GOOGLEVECTORS/"
google_model = KeyedVectors.load_word2vec_format(the_path_in + "GoogleNews-vectors-negative300.bin.gz", binary=True) 




google_terrorist_list = [i[0] for i in google_model.most_similar("terrorist", topn = 15)]
google_terrorist_list

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
