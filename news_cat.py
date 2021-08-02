# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:49:05 2021

@author: Pavithra
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
import gensim
from gensim import corpora, models
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from preprocess import preprocess_text






df = pd.read_excel(r"DataNLP.xlsx")
news_df = pd.DataFrame(df, columns= ['Title','Summary'])

news_df = news_df.fillna("")

news_df['News'] = news_df['Title']+ ". "+ news_df['Summary']
news_df.drop_duplicates(subset ="News", inplace = True)


news_df["News_clean"] = news_df["News"].apply(lambda x: preprocess_text(x))

processed_docs = []

for doc in news_df['News_clean']:
    processed_docs.append(doc)

dictionary = corpora.Dictionary(processed_docs)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(doc_term_matrix)
doc_tfidf = tfidf[doc_term_matrix]

coherence = []
for k in range(3,15):
    print('Round: '+str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_tfidf, num_topics=k, id2word = dictionary,passes=4,eval_every = None)
    
    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=processed_docs,\
                                                      dictionary=dictionary, coherence='c_v')
    coherence.append((k,cm.get_coherence()))


x_val = [x[0] for x in coherence]
y_val = [x[1] for x in coherence]


plt.plot(x_val,y_val)
plt.scatter(x_val,y_val)
plt.title('Number of Topics vs. Coherence')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence')
plt.xticks(x_val)
plt.show()



#Number of Topics selected - 6 (2nd option 3)

Lda = gensim.models.ldamodel.LdaModel
ldamodel_tfidf = Lda(doc_tfidf, num_topics=6, id2word = dictionary, passes=4, eval_every = None, random_state=0)

pickle.dump(ldamodel_tfidf, open('ldamodel_tfidf.pkl', 'wb'))
pickle.dump(dictionary, open('dictionary.pkl', 'wb'))
    

"""
topic_data =  pyLDAvis.gensim_models.prepare(ldamodel_tfidf, doc_tfidf, dictionary, mds = 'pcoa')
pyLDAvis.display(topic_data)


all_topics = {}
num_terms = 10 
lambd = 0.6
for i in range(1,7): 
    topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic'+str(i)].copy()
    topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
    all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values

a = pd.DataFrame(all_topics).T
"""


#topic word clouds
topic = 0 
while topic < 6:
    topic_words_freq = dict(ldamodel_tfidf.show_topic(topic, topn=25))   
    #Word Cloud for topic using frequencies
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq) 
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Category #"+str(topic+1)+"\n",fontsize='22')
    plt.tight_layout()
    plt.savefig("Category"+str(topic+1)+".png")
    topic += 1 
