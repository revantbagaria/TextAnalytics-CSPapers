import numpy as np
import findThePapers, generate_text_list, feature_extractor
from generate_titles import generate_titles
from display_features import display_features
import csv

def tfidf_stats(query_tfidf_features, titles_query):

	query_tfidf_features = query_tfidf_features.toarray()
	filename = 'CIDR_VEE_HCW_combined.csv'
	with open(filename, 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(["Names"]+titles_query)
		
		mean = []
		for i, each in enumerate(query_tfidf_features):
			print(mean.append(np.ndarray.mean(query_tfidf_features[i])))

		csvwriter.writerow(["Means"]+mean)

		minimum = []
		for i, each in enumerate(query_tfidf_features):
			print(minimum.append(np.ndarray.min(query_tfidf_features[i])))

		csvwriter.writerow(["Minimum"]+minimum)

		maximum = []
		for i, each in enumerate(query_tfidf_features):
			print(maximum.append(np.ndarray.max(query_tfidf_features[i])))

		csvwriter.writerow(["Maximum"]+maximum)

		std = []
		for i, each in enumerate(query_tfidf_features):
			print(std.append(np.ndarray.std(query_tfidf_features[i])))

		csvwriter.writerow(["Std"]+std)


def compute_cosine_similarity(doc_features, corpus_features):
	indices_returned = []
	similarity = np.dot(doc_features, corpus_features.T.toarray())
	indices_returned.extend(range(1, len(corpus_features.toarray()) + 1))
	return similarity, indices_returned

def findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name=None):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, matrix = [], []

	for index, doc_tfidf in enumerate(query_tfidf_features):
		similarities, _ = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features)
		matrix.append(similarities)

		# indices = [i for i in range(len(similarities))]
		# res = zip(similarities, indices)

		# for i, each in enumerate(res):
		# 	if abs(each[0]-1.00) <= 0.01:
		# 		del res[i]
		# 		break

		# # print index, similarities
		# res.sort(reverse=True)

		# print("The top 3 most similar documents for " + titles_query[index] + ":")
		# for i in range(3):
		# 	if i < len(res):
		# 		print(titles_corpus[res[i][1]] + ":" + str(res[i][0]))

	display_features(matrix, titles_corpus, titles_query)



def findSummarySimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name=None):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, indices = [], []
	matrix = []
	count = 1

	f = open("intraconf_similarity_stats.txt","a+")

	f.write("Similarity Statistics for {}:\n".format(name))
	f.write("No of papers: %d\n" % len(titles_corpus))

	for index in range(len(query_tfidf_features)):
		doc_tfidf = query_tfidf_features[index]

		similarities, indices_returned = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features[(index+1):])
		matrix.append(np.array([""]*count + list(similarities)))
		count += 1

		for i in range(len(indices_returned)):
			value = indices_returned[i] + index
			indices.append((titles_corpus[index], titles_corpus[value]))

		result.extend(similarities)

	index_max = indices[np.argmax(result)]
	index_min = indices[np.argmin(result)]

	f.write("Mean: %f \n" %np.mean(result))
	f.write("Max: %f, %s \n" % (np.max(result), index_max,))
	f.write("Min: %f, %s \n" % (np.min(result), index_min,))
	f.write("Standard Deviation: %f \n" %np.std(result))
	f.write("\n")

	# display_features(matrix, titles_corpus, titles_query)
	f.close()


# def delete_empty_ones(corpus_docs_extended):

# 	empty_files = ["CCGRID_17_012", "ISPASS_17_007", "Middleware_17_019", "NSDI_17_009", "NSDI_17_037", "NSDI_17_034", "NSDI_17_040", "NSDI_17_042", "NSDI_17_036", "NSDI_17_038", "NSDI_17_035", "NSDI_17_032", "NSDI_17_039", "NSDI_17_041", "NSDI_17_030", "NSDI_17_031", "NSDI_17_024", "NSDI_17_033", "ICPE_17_022"]

# 	for i, each1 in enumerate(corpus_docs_extended):
# 		if isinstance(each1, list):
# 			for j, each2 in enumerate(each1):
# 				if each2 in empty_files:
# 					del each1[j]
# 		else:
# 			if each1 in empty_files:
# 				del corpus_docs_extended[i]

# 	return corpus_docs_extended


def generate_similarity(corpus_docs, query_docs, corpus_docs_append, query_docs_append, same, name=None):

	if not corpus_docs:
		corpus_docs = corpus_docs_append

	if not query_docs:
		query_docs = query_docs_append

	if (not query_docs) and (not corpus_docs):
		print("Please enter both, corpus_docs and query_docs.")
		exit(1)

	corpus_docs_extended = findThePapers.findThePapers(corpus_docs)
	# corpus_docs_extended = delete_empty_ones(corpus_docs_extended)
	titles_corpus = generate_titles(corpus_docs_extended)
	norm_corpus = generate_text_list.generate_text_list(corpus_docs_extended)
	corpus_tfidf_vectorizer, corpus_tfidf_features = feature_extractor.build_feature_matrix(norm_corpus,
														feature_type='tfidf',
														ngram_range=(1, 1), 
														min_df=0.0, max_df=1.0)
	query_docs_extended = findThePapers.findThePapers(query_docs)
	# query_docs_extended = delete_empty_ones(query_docs_extended)
	titles_query = generate_titles(query_docs_extended)
	norm_query = generate_text_list.generate_text_list(query_docs_extended)
	query_tfidf_features = corpus_tfidf_vectorizer.transform(norm_query)

	# tfidf_stats(query_tfidf_features, titles_query)

	print 'Document Similarity Analysis using Cosine Similarity'
	print '='*60

	if same:
		findSummarySimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name)
	else:
		findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name)


