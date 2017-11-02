import text_normalization, text_similarity, extract_from_xml, construct_wordcloud, feature_extractor
import numpy as np
import argparse
import matplotlib.pyplot as plt



def main(combined, individual, nameOfCombined, toy_corpus, query_docs, query_docs_combined1, query_docs_combined2, query_docs_combined3):
	
	count = 1

	if combined:
		combined_text = ""

		for each_paper in combined:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			combined_text += extract_from_xml.extract_from_xml(each_paper)

		combined_text_list = text_normalization.process_text(combined_text)
		construct_wordcloud.wordcloud(combined_text_list, nameOfCombined, count)
		# plt.figure(count)
		# wordcloud = WordCloud().generate(' '.join(combined_text_list))
		# plt.imshow(wordcloud, interpolation='bilinear')
		# plt.axis("off")
		# plt.savefig('/Users/Revant/Desktop/WordClouds/' + nameOfCombined)
		count += 1

	if individual:
		array_strings = []
		individual_names = []

		for each in individual:
			index = each.find('.')
			individual_names.append(each[:index] + ".png")


		for each_paper in individual:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml.extract_from_xml(each_paper)
			individual_text_list = text_normalization.process_text(individual_text)
			array_strings.append(' '.join(individual_text_list))


		for i in range(len(array_strings)):
			construct_wordcloud.wordcloud(array_strings[i], individual_names[i], count)
			
			# plt.figure(count)
			# wordcloud = WordCloud().generate(array_strings[i])
			# # plt.subplot(211)
			# plt.imshow(wordcloud, interpolation='bilinear')
			# plt.axis("off")

			# # wordcloud = WordCloud(max_font_size=60).generate(each)
			# # plt.subplot(212)
			# # plt.imshow(wordcloud, interpolation="bilinear")
			# # plt.axis("off")

			# plt.savefig("/Users/Revant/Desktop/" + "WordClouds/" + individual_names[i])
			count += 1

		bow_vectorizer, bow_features = feature_extractor.bow_extractor(array_strings, (1, 1))
		# features = bow_features.todense()
		# display_features(features, bow_vectorizer.get_feature_names())
		transformer, tfidf_matrix = feature_extractor.tfidf_transformer(bow_features.todense())



	#cosine similarity

	if toy_corpus:
		norm_corpus = []
		for each_paper in toy_corpus:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml.extract_from_xml(each_paper)
			individual_text_list = text_normalization.process_text(individual_text)
			norm_corpus.append(' '.join(individual_text_list))

		tfidf_vectorizer, tfidf_features = feature_extractor.build_feature_matrix(norm_corpus,
                                                        feature_type='tfidf',
                                                        ngram_range=(1, 1), 
                                                        min_df=0.0, max_df=1.0)

	norm_query_docs = []

	if query_docs_combined1:
		s = ""
		for each_paper in query_docs_combined1:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml.extract_from_xml(each_paper)
			individual_text_list = text_normalization.process_text(individual_text)
			s += ' '.join(individual_text_list)
		norm_query_docs.append(s)

	if query_docs_combined2:
		s = ""
		for each_paper in query_docs_combined2:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml.extract_from_xml(each_paper)
			individual_text_list = text_normalization.process_text(individual_text)
			s += ' '.join(individual_text_list)
		norm_query_docs.append(s)

	if query_docs_combined3:
		s = ""
		for each_paper in query_docs_combined3:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml.extract_from_xml(each_paper)
			individual_text_list = text_normalization.process_text(individual_text)
			s += ' '.join(individual_text_list)
		norm_query_docs.append(s)

	if query_docs:
		norm_query_docs = []
		for each_paper in query_docs:
			each_paper = '/Users/Revant/Desktop/ThesisCSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml.extract_from_xml(each_paper)
			individual_text_list = text_normalization.process_text(individual_text)
			norm_query_docs.append(' '.join(individual_text_list))

		query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)

	print 'Document Similarity Analysis using Cosine Similarity'
	print '='*60

	# for index, doc in enumerate(query_docs):
	    
	#     doc_tfidf = query_docs_tfidf[index]
	#     top_similar_docs = text_similarity.compute_cosine_similarity(doc_tfidf,
	#                                              tfidf_features,
	#                                              top_n=2)
	#     print 'Document',index+1 ,':', doc
	#     print 'Top', len(top_similar_docs), 'similar docs:'
	#     print '-'*40 
	#     for doc_index, sim_score in top_similar_docs:
	#         print 'Doc num: {} Similarity Score: {}\nDoc: {}'.format(doc_index+1,
	#                                                                  sim_score,
	#                                                                  toy_corpus[doc_index])  
	#         print '-'*40       
	#     print 

	def mean(arr):
		s = 0

		for i in range(len(arr)):
			s += arr[i]

		return (s/len(arr))

	result = []
	for index, doc in enumerate(query_docs):
	    
	    doc_tfidf = query_docs_tfidf[index]
	    result.extend(text_similarity.compute_cosine_similarity(doc_tfidf,
	                                             tfidf_features,
	                                             top_n=2))

	result = np.array(result)
	print("Mean of CIDR*cerxml: %d" % np.mean(result, dtype=np.float64))
	print("Max of CIDR*cerxml: %d" % np.max(result))
	print("Min of CIDR*cerxml: %d" % np.min(result))
	print("Standard Deviation of CIDR*cerxml: %d" % np.std(result))


	# result = []
	# for index, doc in enumerate(query_docs):
	    
	#     doc_tfidf = query_docs_tfidf[index]
	#     result.append(text_similarity.compute_cosine_similarity(doc_tfidf,
	#                                              tfidf_features,
	#                                              top_n=2))
	
	# result = np.matrix(result)

	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1)
	# ax.set_aspect('equal')
	# plt.imshow(result, interpolation='nearest', cmap=plt.cm.ocean)
	# plt.colorbar()
	# plt.show()




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--combined', nargs = '+', default = None)
	parser.add_argument('--nameOfCombined', default = None)
	parser.add_argument('--individual', nargs = '+', default = None)
	parser.add_argument('--toy_corpus', nargs = '+', default = None)
	parser.add_argument('--query_docs', nargs = '+', default = None)
	parser.add_argument('--query_docs_combined1', nargs = '+', default = None)
	parser.add_argument('--query_docs_combined2', nargs = '+', default = None)
	parser.add_argument('--query_docs_combined3', nargs = '+', default = None)
	args = parser.parse_args()

	# if (not args.combined) and (not args.individual):
	# 	print("Sorry, too few arguments inputted.")
	# 	exit(1)

	# if (args.combined and not args.nameOfCombined):
	# 	print("Please provide name for the combined plot.")

	main(args.combined, args.individual, args.nameOfCombined, args.toy_corpus, args.query_docs, args.query_docs_combined1, args.query_docs_combined2, args.query_docs_combined3)