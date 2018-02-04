import text_normalization, generate_similarity, extract_from_xml, generate_wordcloud, feature_extractor, globalVariables
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main(wordcloud, wordcloud_combfiles, wordcloud_indfiles, similarity, corpus_docs, query_docs, corpus_docs_append, query_docs_append, SummarySimilarities):

	globalVariables.init()

	if wordcloud:
		generate_wordcloud.generate_wordcloud(wordcloud_combfiles, wordcloud_indfiles)

	if similarity:
		generate_similarity.generate_similarity(corpus_docs, query_docs, corpus_docs_append, query_docs_append, SummarySimilarities)

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--wordcloud', default = False)
	parser.add_argument('--wordcloud_combfiles', nargs = '+', default = None, action = 'append')
	parser.add_argument('--wordcloud_indfiles', nargs = '+', default = None)
	parser.add_argument('--similarity', default = False)
	parser.add_argument('--corpus_docs_append', nargs = '+', default = None, action = 'append')
	parser.add_argument('--query_docs_append', nargs = '+', default = None, action = 'append')
	parser.add_argument('--corpus_docs', nargs = '+', default = None)
	parser.add_argument('--query_docs', nargs = '+', default = None)
	parser.add_argument('--SummarySimilarities', default = False)
	args = parser.parse_args()

	main(args.wordcloud, args.wordcloud_combfiles, args.wordcloud_indfiles, args.similarity, args.corpus_docs, args.query_docs, args.corpus_docs_append, args.query_docs_append, args.SummarySimilarities)