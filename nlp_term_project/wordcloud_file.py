### NLP TERM PROJECT - DEREE
### TRAFALIS PANAGIOTIS
### IOANNIS FITSOPOULOS
### EIRINI NOMIKOU
# ---------------------------------


import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.util import ngrams
from collections import defaultdict
import functions



def replace_dot(content):
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
    print(len(content))
    stops = stopwords.words('english')
    new_words = [word.lower() for word in word_tokenize(content.lower()) if not word.lower() in stops]
    # 
    # new_words2 = [word.lower() for str(word) in word_tokenize(new_words) if not word.lower() in stopwords2]
    # newwords2 = [word for word in new_words if not word in stopwords2]
    new_sent = (" ").join(new_words)
    print(len(new_sent))
    return new_sent
        

def remove_punctuations(content):
    puncuations = [ ':',
                    '(', ')', '*', '?',
                    '.', ',', '%', '^',
                    '&', '@', '[', ']',
                    '<', '>', '-', '/',
                    '\\', "'", '"', '!',
                    "#", '$', '+', ';',
                    '<', '>', '=', '@',
                    '[', ']', '_', '|',
                    '{', '}', '~', '΄',
                    '`', '§', '±', '€',
                    "‘","“","’","”"]
    for pnc in puncuations:
        content = content.replace(pnc,"")
    return content

def produce_word_cloud(content):
    nltk.download("stopwords")
    content = content.lower()
    unigrams = defaultdict(lambda:0)
    content = remove_punctuations(content)
    # conent = remove_stopwords(content)
    conent = functions.replace_contractions(content)
    conent = functions.replace_dot(content)
    words = [word.strip().lower() for word in word_tokenize(content)]
    stops = []
    stopwords2 = open("data/stop_words.txt").readlines()
    for w in stopwords2:
        w = w.strip().lower()
        stops.append(w)
    for word in words:
        word = word.strip().lower()
        if not word in stops:
            unigrams[word]+=1
        

    unigrams = dict(unigrams)
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white'
                    ).generate_from_frequencies(unigrams)
    
    # for word in stopwords2:
    #     word = word.lower()
    #     try:
    #         del unigrams[word.strip().lower()]
    #     except KeyError:
    #         pass
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('wordcloud.png')
    


# print(type(stopwords2))

if __name__ == '__main__':
    df = pd.read_csv('data/crypto_news.csv', error_bad_lines=False)

    print(df.url)


    df = pd.read_csv('data/cointelegraph_news_content.csv', error_bad_lines=False)
    df=df.dropna()
    unigrams = defaultdict(lambda:0)
    big_article = ''
    for article in df.content:
        article = article.lower()
        article = replace_dot(article)
        article = replace_contractions(article)
        article = remove_stopwords(article)
        article = remove_punctuations(article   )
        big_article = big_article + str(article) + " "

    words = [word for word in word_tokenize(big_article)]
    for word in words:
        unigrams[word]+=1

    print(unigrams)
    for key,value in unigrams.items():
        print(key,value)

    print(type(unigrams))
    unigrams = dict(unigrams)
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white'
                    ).generate_from_frequencies(unigrams)

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig('output.png')
    plt.show()


    df = pd.read_csv('data/crypto_news.csv', error_bad_lines=False)

    print(df.url)



