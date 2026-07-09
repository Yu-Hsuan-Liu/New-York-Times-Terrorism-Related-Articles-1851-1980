# -*- coding: utf-8 -*-
"""
Final Revised Functions (2025): Consolidated, Cleaned, and Rigorous.
- Removed duplicate/flawed functions (e.g., old check_dictionary, stem_fun).
- Retained only modern, corrected methods (lemmatization, handle_klan_merger, Procrustes).
"""
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from gensim.models import Word2Vec 
import sys
import warnings

# --- 1. Preprocessing Functions (CORRECTED) ---

def clean_text(txt_in):
    """Removes non-alphanumeric characters and cleans text."""
    clean = re.sub('[^A-Za-z0-9]+', " ", txt_in).strip()
    return clean

def rem_sw(var):
    """Removes standard NLTK English stopwords."""
    sw = set(nltk.corpus.stopwords.words('english'))
    my_test = [word for word in var.split() if word not in sw]
    my_test = ' '.join(my_test)
    return my_test

def handle_klan_merger(var):
    """
    REPLACES dictionary filtering (check_dictionary). Retains all words, merges KKK variants.
    """
    fin_txt = []
    for word in var.split():
        word_lower = word.lower()
        if word_lower in ["klux", "kukluxism", "kuklu", "kukiu", "kutlux", "klansman"]:
             fin_txt.append("klan")
        else:
             fin_txt.append(word_lower) 
    return ' '.join(fin_txt)


def check_additional_sw(var):
    """Removes custom, domain-specific stopwords."""
    additional_sw = ["new", "de", "work", "time", "permiss", "one", "would", "the", "The",
                     "two", "three", "year", "work", "span", "man", "men",
                     "r", "e", "p", "copyright", "index", "n", "a", "b", "c",
                     "d", "f", "g", "h", "i", "j","k","l","m","n","o","q",
                     "s","t","u","v","w",'x','y','z', "said", "could", "upon", "ist",
                     "may", "without", "made", "make", "through", "saying",
                     "even","New", "York", "Times", "copyright", "said", "would", "one" ,"two", "year", "mr", "added", "including", "years",
                     "monday", "see", "make", "three", "since", "say", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                     "made", "want", "many", "without", "need", "get", "way", "through", "whether", "next", "used", "saying", "going",
                     "even", "month", "week", "led", "also", "go", "still", "told", "may",
                     "back", "like", "news", "people", "take", "never", "often", "every", "though", "later", "recent", "use", "could", "much",
                     "last", "around", "put", "away", "began", "left", "four", "five", "six", "seven", "eight", "nine", "ten", "good", "months",
                     "accross", "others", "first", "second", "third", "ms", "already", "long", "known", "wrote", "among", "asked", "accross",
                     "according", "might", "high", "yesterday", "today", "tomorrow", "state",
                     "along", "must", "went", "little", "nan", "published", "oe", "mrs",
                     "e", "ay", "te", "ta", "et", "ad", "however", "other", "another", "en", "os",
                     "ar", "shall", "less", "more", "great", "well", "large", "old", "yet", "nothing",
                     "work", "upon", "several", "whose", "matter", "es", "ever", "almost",
                     "know", "ago", "cannot", "thing", "cause", "no", "yes", "er",
                     "span", "index", "newspapers", "1923", "current", "reproduction", "reproduced", "1923current", "newyork", "prohibited",
                     "city", "country", "county", "day", "time", "either", "pg", "owner", "states", "united", "us", "reproduc", "reproduct", "word",
                     "office", "official", "name", "unit", "file", "permission", "january",
                     "february", "march", "april", "may", "june", "july", "august", "september",
                     "october", "november", "december"]
    
    fin_txt = [word for word in var.split() if word not in additional_sw]
    fin_txt = ' '.join(fin_txt)
    return fin_txt

def num_tokens(var):
    """Calculates number of tokens in a string."""
    return len(var.split())

def lem_fun(var):
    """REPLACES STEMMING: Uses WordNet Lemmatizer for better semantic preservation."""
    lemmatizer = WordNetLemmatizer()
    tmp_txt = [lemmatizer.lemmatize(word) for word in var.split()]
    tmp_txt = ' '.join(tmp_txt)
    return tmp_txt

# --- 2. Descriptive/Similarity Helpers ---

# --- File: my_functions_NYT_gpu.py ---

def give_year_labels(df):
    import numpy as np # Ensure this import is still handled correctly either here or globally
    conditions = [
        ((df["year"] >= 1851) & (df["year"] <= 1900)),
        ((df["year"] >= 1901) & (df["year"] <= 1930)),
        ((df["year"] >= 1931) & (df["year"] <= 1950)),
        ((df["year"] >= 1951) & (df["year"] <= 1960)),
        ((df["year"] >= 1961) & (df["year"] <= 1980))]
    values = ['1851-1900', '1901-1930', '1931-1950', '1951-1960', '1961-1980']
    
    # FIX: Add default parameter, ensuring all resulting values are strings
    df['label'] = np.select(conditions, values, default='Unknown') 
    
    return df



def get_token_in_years(df, token_column, label_in):
    tokenlist = []
    for i in df[df.label == label_in][token_column]:
        tokenlist.extend(i.split(' '))
    return(tokenlist)

def token_to_probability_table(token_list):
    total_counts = sum(FreqDist(token_list).values())
    temp_pd = pd.DataFrame()
    temp_pd = temp_pd._append({"probability(%)":"", "raw_frequency":"", "token":"", }, ignore_index = True)
    temp_pd = temp_pd[["token", "raw_frequency", "probability(%)"]]
    for k, l in FreqDist(token_list).items():
        dict1 = {"probability(%)":l*100/total_counts, "raw_frequency":l, "token":k}
        temp_pd = pd.concat([temp_pd, pd.DataFrame.from_dict(dict1, orient = "index").T])
    temp_pd = temp_pd.iloc[1::]
    temp_pd = temp_pd.sort_values(by=['probability(%)'], ascending=False)
    return(temp_pd)

def word_cloud_save(table, pic_name, numbers_of_words):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    d = {}
    for a, x in zip(table.head(numbers_of_words)["token"],
                    table.head(numbers_of_words)["raw_frequency"]):
        d[a] = x
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(pic_name+".PNG")
    plt.show()


# --- 3. Core Modeling Functions ---

def my_vec_fun(df_in, m, n, path_o):
    """Vectorization using CountVectorizer (CPU-stable)."""
    vectorizer = CountVectorizer(ngram_range=(m, n))
    my_vec_t = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_vec_t.columns = vectorizer.get_feature_names_out()
    pickle.dump(vectorizer, open(path_o + "vectorizer.pkl", "wb" ))
    pickle.dump(my_vec_t, open(path_o + "my_vec_data.pkl", "wb" ))
    return my_vec_t

def my_model_fun_grid(df_in, label_in, path_o):
    """
    Random Forest Classifier with GridSearchCV.
    REVISED: Returns X_test, y_test, and y_pred for external validation/metrics.
    """
    from sklearn.model_selection import GridSearchCV
    
    my_model = RandomForestClassifier(random_state=123)
    
    parameters = {'n_estimators': [10, 100], 'max_depth': [None, 1, 10],
                  'random_state': [123], 'criterion': ("gini", "entropy")}
    my_grid = GridSearchCV(my_model, parameters)
    
    # 80/20 train, test, split
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=0.20, random_state=123)
    
    my_grid.fit(X_train, y_train)
    
    print ("Best Score:", my_grid.best_score_)
    best_params = my_grid.best_params_
    
    my_model_opt = RandomForestClassifier(**best_params)
    my_model_opt.fit(X_train, y_train)
    
    pickle.dump(my_model_opt, open(path_o + "my_model.pkl", "wb"))
    
    y_pred = my_model_opt.predict(X_test)
    
    model_metrics = pd.DataFrame(precision_recall_fscore_support(
    y_test, y_pred, average='weighted'))
    model_metrics.index = ["precision", "recall", "fscore", "none"]
    
    the_preds = pd.DataFrame(my_model_opt.predict_proba(X_test))
    the_preds.columns = my_model_opt.classes_

    # REVISED OUTPUT
    return my_model_opt, model_metrics, the_preds, X_test, y_test, y_pred


def score_text(sample_text_i, the_path_i):
    """Predicts the period label for a sample text using the trained model."""
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    # Use the corrected preprocessing pipeline for consistency
    clean_text_i = clean_text(sample_text_i)
    clean_text_i = rem_sw(clean_text_i)
    clean_text_i = handle_klan_merger(clean_text_i) 
    clean_text_i = check_additional_sw(clean_text_i)
    clean_text_i = lem_fun(clean_text_i) # Use lemmatization
    
    vec_model = pickle.load(open(the_path_i + "vectorizer.pkl", "rb"))
    clean_text_vec = vec_model.transform([clean_text_i]).toarray()
    
    my_model = pickle.load(open(the_path_i + "my_model.pkl", "rb"))
    
    the_pred = my_model.predict(clean_text_vec)
    the_pred_proba = np.max(np.array(my_model.predict_proba(clean_text_vec)))
    
    print ("sample_text predicted as", the_pred[0],
            "with likelihood of", the_pred_proba)
    return the_pred, the_pred_proba

# --- 4. Word Embedding and Alignment Functions (Diachronic Rigor) ---

def orthogonal_procrustes(source_vectors, target_vectors):
    """
    Implementation of Orthogonal Procrustes for diachronic embedding alignment.
    Finds the optimal rotation matrix W that aligns source_vectors to target_vectors.
    """
    M = target_vectors.T @ source_vectors
    U, S, V_T = np.linalg.svd(M)
    W = U @ V_T
    return W

def extract_embeddings_domain(df_in, num_vec_in, path_in, filename, sg=0):
    """
    Trains Gensim Word2Vec model for a single time period.
    """
    model = Word2Vec(
        df_in.str.split(), min_count=5, seed = 123,
        vector_size=num_vec_in, workers=1, window=5, sg=sg)
    
    return model

def top_similar_words(df, column_used, start_year, end_year, keyword, the_path, sg = 0, alignment_matrix=None):
    """
    Calculates top similar words for a keyword.
    """
    lemmatizer = WordNetLemmatizer()
    lem_keyword = lemmatizer.lemmatize(keyword)
    my_pd_sub = df[(start_year <= df.year) & (df.year <= end_year)]
    
    # Train the Word2Vec model for the specific period
    stem_model = extract_embeddings_domain(
        my_pd_sub[column_used], 300, the_path, f"{column_used}_{start_year}_{end_year}", sg)
    
    pickle.dump(stem_model, open(the_path + f"{column_used}_{start_year}_{end_year}_embeddings_domain_model{sg}.pkl", "wb"))
    
    # Check if keyword is in the vocabulary before proceeding
    if lem_keyword in stem_model.wv.index_to_key:
        stem_top_words_values = pd.DataFrame(
            [i for i in stem_model.wv.most_similar(lem_keyword, topn=500)],
            columns =[f'{lem_keyword}_lem', f'{lem_keyword}_Values_lem'])
    else:
        stem_top_words_values = pd.DataFrame(
            columns =[f'{lem_keyword}_lem', f'{lem_keyword}_Values_lem'])
        
    return stem_top_words_values