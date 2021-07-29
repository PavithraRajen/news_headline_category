# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 23:02:26 2021

@author: Pavithra
"""

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
day_month_stopwords=['monday','tuesday','wednesday','thursday','friday','saturday','sunday','january','february','march', 'april','may','june','july','august','september','october','november','december']
stop_words = stop_words.union(day_month_stopwords)




def preprocess_text(text):
    #convert to lowercase and remove punctuations, numbers and characters and then strip
    text = re.sub(r'[^a-z\s]', '', str(text).lower().strip())
    
    # Tokenize 
    clean_text = word_tokenize(text)
    
    # remove Stopwords
    clean_text = [w for w in clean_text if w not in stop_words]
                
    # Lemmatisation (convert the word into root word)
    clean_text = [lemma.lemmatize(w,pos = 'v') for w in clean_text]
            
    return clean_text
