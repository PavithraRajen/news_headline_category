# Categorization of News Article

#### Task - Classify input news article and map it to an appropriate category by applying Natural Language Processing techniques in the given dataset.

## Dataset
Summary of news headlines of US stock market; it contains 1554 news summary. 

  Metadata:
  1.	Title – contains the title of the article
  2.	Summary – contains the summary of the article 
  3.	Published On - contains date in which the article is published


## Approach

The Title and Summary are joined to create the variable News. There was duplicates of News which was removed. 

### Data Pre-processing
- Convert to lowercase and remove punctuations, numbers and characters.
- Tokenization
- Remove Stopwords (Days and Months are added to the nltk stop words)
- Words are lemmatized.

### LDA

By using **Topic modeling** an ‘unsupervised’ machine learning technique, we will classify the news articles into different categories. Here we are going to apply Latent Dirichlet Allocation(LDA) to cluster together similar documents by similar topics.

We use the gensim LDA model. The two main inputs to the LDA topic model are the dictionary and the corpus. 
**Gensim doc2bow** - For each document we create a dictionary(words and how many times those words appear)

#### LDA using TF-IDF
We create tf-idf model object using models and train our lda model using gensim.models.ldamodel.LdaModel. 

### Choosing the Number of Topics for LDA

To find number of topics we have to build LDA models with different values of k (no. of topics) and pick the one that gives the highest coherence value. 

We found the best value of k where max topic coherence is 6.

The final LDA model is built with **number of topics - 6**


### Workable app

The FLASK app is deployed on Heroku. 

![NewsCategory](https://github.com/PavithraRajen/news_headline_category/blob/main/static/img/app.PNG)

Find the app here - **https://news-topics.herokuapp.com/**

















