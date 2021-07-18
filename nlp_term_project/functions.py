### NLP TERM PROJECT - DEREE
### TRAFALIS PANAGIOTIS
### IOANNIS FITSOPOULOS
### EIRINI NOMIKOU
# ---------------------------------


from nltk import sent_tokenize, word_tokenize
import nltk
import pandas as pd
import re
import spacy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
import datetime
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords


nltk.download("stopwords")
nlp = spacy.load('en_core_web_sm')
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
# sbert_model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
steps = """
1) replace puncuations without space if needed and replace contractions
2) sentence tokenize in each article
3) vectorize each sentence (using bert)
4) clustering at sentences (using kmeans)
5) find cluster center
6) keep the sentences closer to the center of each cluster
7) extract sentences

"""



def replace_dot(content):
    # takes as input an article and returns the same content,
    # but with one space character added if needed between sentences
    # in order to help nltk.tokenize produce better results
    criterion = r'[A-Za-z0-9][A-Za-z0-9][\.\!\?][A-Za-z][A-Za-z]*'
    
    outcome = re.findall(criterion,content)
    if len(outcome) > 0:
        for item in outcome:
            new_item1 = item.replace( ".", ". " )
            new_item2 = item.replace( "?", "? " )
            new_item3 = item.replace( "!", "! " )

            content = content.replace(item, new_item1)
            content = content.replace(item, new_item2)
            content = content.replace(item, new_item3)
    

    return content


def replace_contractions(content):
    # takes as input a string and returns the 
    # string with replaced contractions
    replacements = {
    "won't":"will not",
    "can't":"can not",
    "i'm":"i am",
    "he's":"he is",
    "she's":"she is",
    "it's":"it is",
    "that's":"that is",
    "here's":"here is",
    "there's":"there is",
    "i've": "I have",
    "won't": "will not",
    "could've": "could have",
    "wouldn't": "would not",
    "it's": "It is",
    "i'll": "I will",
    "haven't": "have not",
    "can't": "can not",
    "that's": "that is",
    "they'r": "they are",
    "doesn't": "does not",
    "don't": "do not",
    "i'm": "I am",
    "story's": "story s",
    "souldn't've": "sould not have",
    "n't":" not",
    "n't":" not",
    "'ll":" will",
    "'ve": " have",
    "'re":" are",
    "'s":" s",
    "’s":" s",
    "'":" "
        
    }
    for key,value in replacements.items():
        content = content.replace(key,value)
    return content 

def remove_stopwords(content):
    # removes stopsowrds from given text uning nltk corpus
    stops = stopwords.words('english')
    new_words = [word for word in word_tokenize(content.lower()) if not word in stops]
    new_sent = (" ").join(new_words)
    return new_sent
        
# df['sentences'] = df['content'].apply(split_into_sentences)

def tokenize_sentences(content):
    # Splits all sentences of a given text and
    # returns a list
    sentences = [sent for sent in sent_tokenize(content)]
    return sentences

def vectorize_sentences(content):
    # Input sentence
    # output vectorized sentence [array with embendings]
    vectorize = nlp(content)
    return vectorize.vector

def bert_embendings(content): 
    # Input sentence
    # output vectorized sentence [array with bert embendings]   
    sentence_embeddings = sbert_model.encode(content)
    return sentence_embeddings

def cluster_sents(df_content,max_clusters=12):
        #  Takes as input all a pd.Series of sentence emendings
        #  Applies kmeans clustering and returns a list of labels
        #  and two lists with clusters and a list with the centers

        # max_clusters = 14
        last_silhouette = 0
        opt_n_clusters = 0

        X = np.array(df_content.tolist())

        if len(X) < max_clusters:
            max_clusters = len(X)

        # print('LENGHT', len(X),'--------////////////////-----------------------')
        for i in range(2,max_clusters):
            kmeans = KMeans(n_clusters=i, random_state = 1)

            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)

            silhouette_values = silhouette_samples(X, y_kmeans)
            silhouette_values = round(np.mean(silhouette_values),3)
            # print(y_kmeans)

            centers = kmeans.cluster_centers_
            # print(centers)
            # print(y_kmeans)

            if silhouette_values > last_silhouette:
                last_silhouette = silhouette_values
                opt_n_clusters = i

        if opt_n_clusters >= 2:
            kmeans = KMeans(n_clusters = opt_n_clusters)
            kmeans.fit(X)
            y_kmeans = kmeans.predict(X)
            centers = kmeans.cluster_centers_
            
            unique_clusters = sorted(list(dict.fromkeys(y_kmeans)))
            # print(unique_clusters)

            cluster_centers = dict(zip(unique_clusters,centers))

            # print('\n\n\nNUMBER OF CLUSTERS: ',opt_n_clusters,'\n\n\n')
            print('---------')
            print(cluster_centers)
            print('--------------')

            return y_kmeans, cluster_centers

def difference(row):
    #type of emb and centroid is different, hence using tolist below
    return distance_matrix([row['Embendings']], [row['Centers'].tolist()])[0][0]

def number_of_sents(number_of_clusters,max_sents_output=24):
    # Input:
    # max number of sents for the summary output (default = 24)
    # number of clusters

    # Output:
    # Returns: How many sentences to be extracted from each cluster (Integer)
    return int(round(max_sents_output/number_of_clusters))
    
def main(article,method = bert_embendings):
    # Not in use
    article = replace_dot(article)
    sentences = [sent for sent in sent_tokenize(article)]
    print('num of sentences of article:',len(sentences))
    # print(article)
    df_article = pd.DataFrame(sentences,columns=['Sentences'])
    
    df_article['Embendings'] = df_article['Sentences'].apply(method)

    clusters, cluster_centers = cluster_sents(df_article['Embendings'],14)

    # print(clusters)
    df_article['Clusters'] = clusters
    print('Number of Clusters:',len(df_article['Clusters'].unique()))
    # print(cluster_centers[1])
    df_article['Centers']=df_article['Clusters'].map(cluster_centers)

    

    df_article['Difference'] = df_article.apply(difference, axis=1)
    # print( df_article['Difference'])

    if len(sentences) < 24:
        num_sent_to_return = number_of_sents(len(df_article['Clusters'].unique()),max_sents_output=len(sentences))
    else:
        num_sent_to_return = number_of_sents(len(df_article['Clusters'].unique()),max_sents_output=24)

    summary=' '.join(df_article.sort_values('Difference',ascending = True).groupby('Clusters').head(num_sent_to_return).sort_index()['Sentences'].tolist())

    return summary

    

    # return summary

if __name__ == '__main__':
    df = pd.read_csv('data/cointelegraph_news_content.csv', error_bad_lines=False)
    df=df.sample(n=30)
    big_article = ''
    for article in df.content:
        big_article = big_article + str(article)
    print(big_article)

    print('\n\n----------------\n\n')
    print(main(big_article))
    text = open("testfile.txt").read()
    text = replace_dot(text)
    text = replace_contractions(text)
    text = remove_stopwords(text)

    # sents=[sent for sent in sent_tokenize(text)]
    # print(len(sents))
    print(text)
