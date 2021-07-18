### NLP TERM PROJECT - DEREE
### TRAFALIS PANAGIOTIS
### IOANNIS FITSOPOULOS
### EIRINI NOMIKOU
# ---------------------------------



import functions
from nltk import sent_tokenize, word_tokenize
import pandas as pd
import math
import keywords
import wordcloud_file

pd.set_option('display.max_columns', None)

# From class summarize_text method single_article can be called in every module:
# Example:
# article = """ ....article.... """
# summarizer = summarize_text(article, max_summary_len=20) 
# print(summarizer.single_article())
#
#
#
# method multiple_articles:
# Example:
# articles = [ """ ....article.... """ , """ ....article2....""", ....., """....article_n....""" ]
# summarizer = summarize_text(articles, max_summary_len=20) 
# print(summarizer.multiple_articles())
#  
# 

class summarize_text:
    def __init__(self,articles,max_summary_len):
        # Input, single article or list of articles
        # and the maximun number of sentences for the summary
        self.article = articles
        self.max_summary_len = max_summary_len



    def single_article(self):
        # Takes input a unique article and returns 
        #it's summary 
        # a word cloud of the article
        # and it's keywords

        wordcloud_file.produce_word_cloud(self.article)
        self.article = functions.replace_dot(self.article)
        self.article = functions.replace_contractions(self.article)

        sentences_raw = [sent for sent in sent_tokenize(self.article)]   

        df_article = pd.DataFrame(sentences_raw,columns=['sentences_raw'])
        print(df_article)
        df_article['Sentences'] = df_article['sentences_raw']
        df_article['Embendings'] = df_article['Sentences'].apply(functions.bert_embendings)

        clusters, cluster_centers = functions.cluster_sents(df_article['Embendings'],14)
        df_article['Clusters'] = clusters
        
        print('num of sentences of article:',len(sentences_raw))
        print('Number of Clusters:',len(df_article['Clusters'].unique()))
        df_article['Centers']=df_article['Clusters'].map(cluster_centers)

        df_article['Difference'] = df_article.apply(functions.difference, axis=1)
        unique_clusters = df_article['Clusters'].unique()
        print(unique_clusters)


        if len(sentences_raw) < 100:
            if len(unique_clusters) <= math.ceil(0.1 * len(sentences_raw)):
                self.max_summary_len = math.ceil(len(unique_clusters))
            else:
                self.max_summary_len = math.ceil(0.1 * len(sentences_raw))
                    
        
        num_sent_to_return = functions.number_of_sents(len(df_article['Clusters'].unique()),max_sents_output=self.max_summary_len)
        print('return sentences:',num_sent_to_return)
        print('max_summary_len:',self.max_summary_len)
        
        summary=' '
        for cluster in unique_clusters:
            print('return',num_sent_to_return,'number of sentences for cluster',cluster)
            x = df_article[df_article['Clusters'] == cluster].sort_values('Difference',ascending = True)['sentences_raw'][:num_sent_to_return]
            for sentence in x:
                summary = str(summary) + str(sentence)

        kw = keywords.keywords(summary)
        return summary, kw


    def multiple_articles(self):
        # Takes as input a list of articles and merge them into one
        # Calls the single_article function and returns its outputs
        big_article = ''
        for article in self.article:
            article = functions.replace_dot(article)
            article = functions.replace_contractions(article)
            big_article = big_article + str(article) + " "

        words = [word for word in word_tokenize(big_article)]
        self.article = big_article
        x = self.single_article()

        return x


if __name__ == '__main__':
    # By runnint this file as it is, it will produce a summary example of the dataframe
    # for the articles of one month
    df = pd.read_csv('data/cointelegraph_news_content.csv', error_bad_lines=False)
    dates= df['date'].str.split('-',expand=True)
    df['year'] = dates[0].astype(str)
    df['month'] = dates[1].astype(str)
    print(df['year'].unique())
    print(df['month'].unique())
    df = df[(df['year']== "2019") & (df['month']== "01")].sample(n=100)
    # df = pd.read_csv('data/cointelegraph_news_content.csv', error_bad_lines=False,nrows=5)
    print(len(df))
    articles = df.content.to_list()
    summary = summarize_text(articles,max_summary_len=20)
    summary_text,kw = summary.multiple_articles()
    print(summary_text)
    print(kw)

    text = open("testfile.txt").read()
    text = functions.replace_dot(text)
    sents=[sent for sent in sent_tokenize(text)]
    print(len(sents))
