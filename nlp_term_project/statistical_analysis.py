### NLP TERM PROJECT - DEREE
### TRAFALIS PANAGIOTIS
### IOANNIS FITSOPOULOS
### EIRINI NOMIKOU
# ---------------------------------


import pandas as pd
from nltk import sent_tokenize
import numpy as np
import matplotlib.pyplot as plt
import re


df = pd.read_csv('data/cointelegraph_news_content.csv', error_bad_lines=False)
# print(df.columns)
print(len(df))
df_2 = pd.read_csv('data/crypto_news.csv')
print(len(df_2))
# print(df_2.columns)
df_2['content'] = df_2['text']

df = df['content']
df_2 = df_2['content']
df = pd.concat([df,df_2])
# df=df.sample(n=10)

def sentences(row):
    try:
        # print(row)
        sents = len(sent_tokenize(row['content']))
        return sents
    except:
        return np.NaN


def scaler(row):
    return (row['sents'] - df['sents'].mean())/df['sents'].std()



def replace_dot(content):
    criterion = r'[A-Za-z0-9][A-Za-z0-9][\.\!\?][A-Za-z][A-Za-z]'
    try:
        outcome = re.findall(criterion,content)
        if len(outcome) > 0:
            for item in outcome:
                content = re.sub(item,item.replace('.','. '),content)
                content = re.sub(item,item.replace('?','? '),content)
                content = re.sub(item,item.replace('!','! '),content)
            # print('replaced dots')
            return content
    except TypeError:
        return content

df=pd.DataFrame(df)

df['content'] = df.apply(replace_dot)


df['sents'] = df.apply(sentences,axis=1)

df = df.dropna()

print('lenght:',len(df))
df['scaled_sents'] = df.apply(scaler,axis=1)


print('------------------Standarized Data------------------------')
print('mean number of sentences:',df['scaled_sents'].mean())
print('median number of sentences:',df['scaled_sents'].median())
print('std number of sentences:',df['scaled_sents'].std())

print('------------------Absolut Data----------------------------')
print('mean number of sentences:',df['sents'].mean())
print('median number of sentences:',df['sents'].median())
print('std number of sentences:',df['sents'].std())
median_ = df['scaled_sents'].median()
plt.figure(1,figsize=(12,8))
df['scaled_sents'].plot.hist()
plt.axvline(x=df['scaled_sents'].median(),color='r')
plt.text(median_,0,'Median',rotation=45,color='r')
plt.title('Number of sentences per article - Distribution (On Scaled Data)')
plt.xlabel('Number of sentences')
plt.savefig('Number of sentences per article - Distribution (On Scaled Data).jpeg')
plt.show()

plt.figure(2,figsize=(12,8))
df['sents'].plot.hist()
plt.axvline(x=df['sents'].median(),color='r')
plt.text(median_,0,'Median',rotation=45,color='r')
plt.title('Number of sentences per article - Distribution')
plt.xlabel('Number of sentences')
plt.savefig('Number of sentences per article - Distribution.jpeg')
plt.show()
