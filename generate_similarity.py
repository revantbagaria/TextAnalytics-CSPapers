import numpy as np
import findThePapers, generate_text_list, feature_extractor

def compute_cosine_similarity(doc_features, corpus_features, top_n=3):
    # get document vectors
    # print("############\n", doc_features)
    # doc_features = doc_features[0]
    # compute similarities
    similarity = np.dot(doc_features, 
                        corpus_features.T.toarray())
    # similarity = similarity.toarray()[0]
    indices_returned = []
    indices_returned.extend(range(1, len(corpus_features.toarray()) + 1))
    return similarity, indices_returned

def findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features):

def findSummarySimilarities(query_tfidf_features, corpus_tfidf_features):

	query_tfidf_features = query_tfidf_features.toarray()
	result, indices = [], []

	for index in range(len(query_tfidf_features)):
		doc_tfidf = query_docs_tfidf[index]
	    similarities, indices_returned = text_similarity.compute_cosine_similarity(doc_tfidf,
                                         tfidf_features[index+1:],
                                         top_n=2)

	    for i in range(len(indices_returned)):
	    	# value = (indices_returned[i] + index) % (len(query_docs_tfidf) - 1)
	    	value = (indices_returned[i] + index)
	    	indices.append((index+1, value+1))

	    result.extend(similarities)


	    # result.extend(text_similarity.compute_cosine_similarity(doc_tfidf,
	    #                                          tfidf_features[index+1:],
	    #                                          top_n=2))

	index_max = indices[np.argmax(result)]
	index_min = indices[np.argmin(result)]

	print("Mean of HCW: %f" %np.mean(result))
	print("Max of HCW: %f, %s" % (np.max(result), index_max,))
	print("Min of HCW: %f, %s" % (np.min(result), index_min,))
	print("Standard Deviation of HCW: %f" %np.std(result))


def generate_similarity(corpus_docs, query_docs, corpus_docs_append, query_docs_append, SummarySimilarities):

	if not corpus_docs:
		corpus_docs = corpus_docs_append

	if not query_docs:
		query_docs = query_docs_append

	if (not query_docs) and (not corpus_docs):
		print("Please enter both, corpus_docs and query_docs.")
		exit(1)

	corpus_docs_extended = findThePapers.findThePapers(corpus_docs)
	norm_corpus = generate_text_list.generate_text_list(corpus_docs_extended)
	corpus_tfidf_vectorizer, corpus_tfidf_features = feature_extractor.build_feature_matrix(norm_corpus,
                                                        feature_type='tfidf',
                                                        ngram_range=(1, 1), 
                                                        min_df=0.0, max_df=1.0)


	query_docs_extended = findThePapers.findThePapers(query_docs)
	norm_query = generate_text_list.generate_text_list(query_docs_extended)
	query_tfidf_features = corpus_tfidf_vectorizer.transform(norm_query)

	print 'Document Similarity Analysis using Cosine Similarity'
	print '='*60

	if SummarySimilarities:
		findSummarySimilarities(query_tfidf_features, corpus_tfidf_features)
	else:
		findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features)


