from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re



def keywords(content):
	n_gram_range = (1, 1)
	stop_words = "english"

	# Extract candidate words/phrases
	count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([content])
	candidates = count.get_feature_names()

	model = SentenceTransformer('distilbert-base-nli-mean-tokens')
	doc_embedding = model.encode([content])
	candidate_embeddings = model.encode(candidates)

	top_n = 5
	distances = cosine_similarity(doc_embedding, candidate_embeddings)
	keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

	return keywords


if __name__=='__main':

	df = pd.read_csv('data/cointelegraph_news_content.csv', error_bad_lines=False,nrows=5)
	print(df.columns)


	df['keywords'] = df['content'].apply(keywords)
	print(df.keywords.tolist())
	# df.to_csv('test.csv')