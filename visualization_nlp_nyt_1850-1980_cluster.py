# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:50:18 2021

@author: tosea
"""
#visualization
import pandas as pd
from gensim.models import Word2Vec
from my_functions_NYT import *
import pickle
import nltk
from gensim.models import KeyedVectors


the_path = "C:/Users/tosea/NYT_API/"



keyword = "terrorist"
topn = 20

word2vec_visual_hamilton(keyword, topn, f"hamilton_{keyword}.png", 
                         stem = True,  google_background = True)

word2vec_visual_cluster(keyword, topn, f"cluster_{keyword}.png",
                        stem = True)
























