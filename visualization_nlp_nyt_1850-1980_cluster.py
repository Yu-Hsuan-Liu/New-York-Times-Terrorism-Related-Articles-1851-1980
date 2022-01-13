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
topn = 12

word2vec_visual_hamilton(keyword, topn, f"hamilton_{keyword}", 
                         stem = True,  google_background = False, sg = 1)

word2vec_visual_cluster(keyword, topn, f"cluster_{keyword}",
                        stem = True, sg = 1)
























