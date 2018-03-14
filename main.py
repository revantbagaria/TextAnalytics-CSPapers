import text_normalization, generate_similarity, generate_classification, extract_from_xml, generate_wordcloud, feature_extractor, globalVariables, generate_cluster
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main(wordcloud, wordcloud_combfiles, wordcloud_indfiles, similarity, corpus_docs, query_docs, corpus_docs_append, query_docs_append, same, name, clustering, cluster_files, knn_clustering, affinity_prop, db, hdb, ward, classification, classification_files, classification_labels):

	globalVariables.init()

	if wordcloud:
		generate_wordcloud.generate_wordcloud(wordcloud_combfiles, wordcloud_indfiles)

	if similarity:
		generate_similarity.generate_similarity(corpus_docs, query_docs, corpus_docs_append, query_docs_append, same, name)

	if clustering:
		generate_cluster.clustering(cluster_files, knn_clustering, affinity_prop, db, hdb, ward)

	if classification:
		generate_classification.generate_classification(classification_files, classification_labels)

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-w', '--wordcloud', default = False)
	parser.add_argument('-wc', '--wordcloud_combfiles', nargs = '+', default = None, action = 'append')
	parser.add_argument('-wi', '--wordcloud_indfiles', nargs = '+', default = None)
	parser.add_argument('-s', '--similarity', default = False)
	parser.add_argument('-cda', '--corpus_docs_append', nargs = '+', default = None, action = 'append')
	parser.add_argument('-qda', '--query_docs_append', nargs = '+', default = None, action = 'append')
	parser.add_argument('-cd', '--corpus_docs', nargs = '+', default = None)
	parser.add_argument('-qd', '--query_docs', nargs = '+', default = None)
	parser.add_argument('-name', '--name', default = None)
	parser.add_argument('-same', '--same', default = False)
	parser.add_argument('-cl', '--clustering', default = False)
	parser.add_argument('-clf', '--cluster_files', nargs = '+', default = None)
	parser.add_argument('-knn', '--knn_clustering', default = None)
	parser.add_argument('-aff', '--affinity_prop', default = None)
	parser.add_argument('-ward', '--ward', default = None)
	parser.add_argument('-db', '--dbscan', nargs = '+', default = None)
	parser.add_argument('-hdb', '--hdbscan', default = None)
	parser.add_argument('-cf', '--classification', nargs = '+', default = None)
	parser.add_argument('-cff', '--classification_files', nargs = '+', default = None)
	parser.add_argument('-cfl', '--classification_labels', nargs = '+', default = None)
	args = parser.parse_args()

	main(args.wordcloud, args.wordcloud_combfiles, args.wordcloud_indfiles, args.similarity, args.corpus_docs, args.query_docs, args.corpus_docs_append, args.query_docs_append, args.same, args.name, args.clustering, args.cluster_files, args.knn_clustering, args.affinity_prop, args.dbscan, args.hdbscan, args.ward, args.classification, args.classification_files, args.classification_labels)