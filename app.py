# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 23:24:09 2021

@author: Pavithra
"""
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from preprocess import preprocess_text
from gensim import corpora, models

ldamodel = pickle.load(open('ldamodel_tfidf.pkl', 'rb'))
dictionary = pickle.load(open('dictionary.pkl', 'rb'))


app = Flask(__name__)



@app.route('/',methods=['GET','POST'])
def predict():
    p = ""
    doc = ""
    
    if request.method == 'POST':
        doc = request.form['message']
        bow_v = dictionary.doc2bow(preprocess_text(doc))
        d_score = sorted(ldamodel[bow_v], key=lambda tup: -1*tup[1])[0][1]
        if d_score >= 0.2:
            p = "Category: "+ str(sorted(ldamodel[bow_v], key=lambda tup: -1*tup[1])[0][0]+1)
        else:
            p = "Other"
    if doc == "":
        p  = "Enter the news..."
    
    return render_template('home.html',prediction = p, document = doc)



if __name__ == '__main__':
	app.run(debug=False)
