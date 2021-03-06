import numpy as np
import findThePapers, generate_text_list, feature_extractor
from generate_titles import generate_titles
from display_features import display_features
import csv
import pandas as pd

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

def all_latest_func(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, dn, name=None):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, matrix = [], []

	for index, doc_tfidf in enumerate(query_tfidf_features):
		similarities, _ = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features)

		indices = [i for i in range(len(similarities))]
		res = zip(similarities, indices)

		for i, each in enumerate(res):
			if abs(each[0]-1.00) <= 0.01:
				del res[i]
				break

		filename = "similarity_half" + str(dn) + ".csv"
		df = pd.read_csv(filename, index_col='Names')
		# df = df.set_index('Names')
		df.loc[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		# df.at[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		df.to_csv(filename, mode='w')

		# df2 = pd.read_csv('similarity_full.csv', index_col='Names')
		# # df = df.set_index('Names')
		# df2.loc[titles_query[index], titles_corpus[res[0][1]]] = res[0][0]
		# df2.loc[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		# df2.to_csv("similarity_full.csv", mode='w')


def intraconf_latest_func(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, filename, name=None):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, matrix = [], []

	for index, doc_tfidf in enumerate(query_tfidf_features):
		similarities, _ = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features)

		indices = [i for i in range(len(similarities))]
		res = zip(similarities, indices)

		for i, each in enumerate(res):
			if abs(each[0]-1.00) <= 0.01:
				del res[i]
				break

		# df = pd.read_csv('similarity_half.csv', index_col='Names')
		# # df = df.set_index('Names')
		# df.loc[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		# # df.at[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		# df.to_csv("similarity_half.csv", mode='w')

		df2 = pd.read_csv(filename, index_col='Names')
		# df = df.set_index('Names')
		df2.loc[titles_query[index], titles_corpus[res[0][1]]] = res[0][0]
		df2.loc[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		df2.to_csv(filename, mode='w')


def combined_latest_func(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name=None):

	query_tfidf_features = query_tfidf_features.toarray()
	result, matrix = [], []

	for index, doc_tfidf in enumerate(query_tfidf_features):
		similarities, _ = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features)

		indices = [i for i in range(len(similarities))]
		res = zip(similarities, indices)

		for i, each in enumerate(res):
			if abs(each[0]-1.00) <= 0.01:
				del res[i]
				break

		df = pd.read_csv('similarity_combined_half.csv', index_col='Names')
		# df = df.set_index('Names')
		df.loc[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		# df.at[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		df.to_csv("similarity_combined_half.csv", mode='w')

		df2 = pd.read_csv('similarity_combined_full.csv', index_col='Names')
		# df = df.set_index('Names')
		df2.loc[titles_query[index], titles_corpus[res[0][1]]] = res[0][0]
		df2.loc[titles_corpus[res[0][1]], titles_query[index]] = res[0][0]
		df2.to_csv("similarity_combined_full.csv", mode='w')


def findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name=None):
	
	query_tfidf_features = query_tfidf_features.toarray()
	result, matrix = [], []

	for index, doc_tfidf in enumerate(query_tfidf_features):
		similarities, _ = compute_cosine_similarity(doc_tfidf, corpus_tfidf_features)
		matrix.append(similarities)

		indices = [i for i in range(len(similarities))]
		res = zip(similarities, indices)

		for i, each in enumerate(res):
			if abs(each[0]-1.00) <= 0.01:
				del res[i]
				break

		# print index, similarities
		res.sort(reverse=True)

		print("The top 3 most similar documents for " + titles_query[index] + ":")
		for i in range(3):
			if i < len(res):
				print(titles_corpus[res[i][1]] + ":" + str(res[i][0]))

	# display_features(matrix, titles_corpus, titles_query)



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

	display_features(matrix, titles_corpus, titles_query)
	f.close()


def generate_similarity(corpus_docs, query_docs, corpus_docs_append, query_docs_append, all_latest, combined_latest, intraconf_latest, filename, same, dn, name=None):

	if not corpus_docs:
		corpus_docs = corpus_docs_append

	if not query_docs:
		query_docs = query_docs_append

	if (not query_docs) and (not corpus_docs):
		print("Please enter both, corpus_docs and query_docs.")
		exit(1)

	if all_latest:
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

		all_latest_func(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, dn, name)	
		return

	if intraconf_latest:
		corpus_docs_extended = corpus_docs
		titles_corpus = generate_titles(corpus_docs_extended)
		norm_corpus = generate_text_list.generate_text_list(corpus_docs_extended)
		corpus_tfidf_vectorizer, corpus_tfidf_features = feature_extractor.build_feature_matrix(norm_corpus,
															feature_type='tfidf',
															ngram_range=(1, 1), 
															min_df=0.0, max_df=1.0)
		
		query_docs_extended = query_docs
		titles_query = generate_titles(query_docs_extended)
		norm_query = generate_text_list.generate_text_list(query_docs_extended)
		query_tfidf_features = corpus_tfidf_vectorizer.transform(norm_query)

		filename = 	filename = filename[:(len(filename)-2)] + ".csv"
		intraconf_latest_func(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, filename, name=None)
		return


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

	if combined_latest:
		combined_latest_func(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name)
		return

	# tfidf_stats(query_tfidf_features, titles_query)

	# print 'Document Similarity Analysis using Cosine Similarity'
	# print '='*60

	# if same:
	# 	findSummarySimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name)
	# else:
	# 	findIndividualSimilarities(query_tfidf_features, corpus_tfidf_features, titles_corpus, titles_query, name)


