import numpy as np
import findThePapers, generate_text_list, feature_extractor
from generate_titles import generate_titles
from display_features import display_features

def compute_cosine_similarity(doc_features, corpus_features):
	indices_returned = []
	similarity = np.dot(doc_features, corpus_features.T.toarray())
	indices_returned.extend(range(1, len(corpus_features.toarray()) + 1))
	return similarity, indices_returned

def findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, matrix = [], []

	for index, doc_tfidf in enumerate(query_tfidf_features):
		similarities, _ = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features)

		matrix.append(similarities)

		indices = [i for i in range(len(similarities))]
		res = zip(similarities, indices)

		for i, each in enumerate(res):
			if abs(each[0]-1.00) <= 0.02:
				del res[i]
				break

		# print index, similarities
		res.sort(reverse=True)

		print("The top 3 most similar documents for " + titles_query[index] + ":")
		for i in range(3):
			if i < len(res):
				print(titles_corpus[res[i][1]] + ":" + str(res[i][0]))

	display_features(matrix, titles_corpus, titles_query)



def findSummarySimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, indices = [], []
	matrix = []
	count = 1
	for index in range(len(query_tfidf_features)):
		doc_tfidf = query_tfidf_features[index]

		similarities, indices_returned = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features[(index+1):])
		matrix.append(np.array([""]*count + list(similarities)))
		count += 1

		for i in range(len(indices_returned)):
			value = indices_returned[i] + index
			indices.append((titles_corpus[index], titles_corpus[value]))
			# indices.append((index+1, value+1))

		result.extend(similarities)

	index_max = indices[np.argmax(result)]
	index_min = indices[np.argmin(result)]

	print("Mean: %f" %np.mean(result))
	print("Max: %f, %s" % (np.max(result), index_max,))
	print("Min: %f, %s" % (np.min(result), index_min,))
	print("Standard Deviation: %f" %np.std(result))

	display_features(matrix, titles_corpus, titles_query)


def generate_similarity(corpus_docs, query_docs, corpus_docs_append, query_docs_append, same):

	if not corpus_docs:
		corpus_docs = corpus_docs_append

	if not query_docs:
		query_docs = query_docs_append

	if (not query_docs) and (not corpus_docs):
		print("Please enter both, corpus_docs and query_docs.")
		exit(1)

	corpus_docs_extended = findThePapers.findThePapers(corpus_docs)
	titles_corpus = generate_titles(corpus_docs_extended)
	norm_corpus = generate_text_list.generate_text_list(corpus_docs_extended)
	corpus_tfidf_vectorizer, corpus_tfidf_features = feature_extractor.build_feature_matrix(norm_corpus,
														feature_type='tfidf',
														ngram_range=(1, 1), 
														min_df=0.0, max_df=1.0)
	query_docs_extended = findThePapers.findThePapers(query_docs)
	titles_query = generate_titles(query_docs_extended)
	norm_query = generate_text_list.generate_text_list(query_docs_extended)
	query_tfidf_features = corpus_tfidf_vectorizer.transform(norm_query)

	print 'Document Similarity Analysis using Cosine Similarity'
	print '='*60

	if same:
		findSummarySimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query)
	else:
		findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query)


