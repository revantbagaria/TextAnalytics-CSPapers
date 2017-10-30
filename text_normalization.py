from contraction_mapping import contraction_dict
import re, nltk, string, sys
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET
from pattern.en import tag
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np


def tokenize_text(text):
	tokens = nltk.word_tokenize(text)
	tokens = [token.strip() for token in tokens]
	return tokens

def expandContractions(text):
    def replace(match):
    	return contraction_dict[match.group(0)]
    c_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return c_re.sub(replace, text)

def pos_tag_text(text):
	def penn_to_wn_tags(pos_tag):
		if pos_tag.startswith('J'):
			return wn.ADJ
		elif pos_tag.startswith('V'):
			return wn.VERB
		elif pos_tag.startswith('N'):
			return wn.NOUN
		elif pos_tag.startswith('R'):
			return wn.ADV
		else:
			return None
	tagged_text = tag(text)
	tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
	return tagged_lower_text


# lemmatize text based on POS tags
def lemmatize_text(text):
	wnl = WordNetLemmatizer()
	pos_tagged_text = pos_tag_text(text)
	lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
	else word
	for word, pos_tag in pos_tagged_text]
	lemmatized_text = ' '.join(lemmatized_tokens)
	return lemmatized_text

def remove_stopwords(tokens):
	stopword_list = nltk.corpus.stopwords.words('english')
	filtered_tokens = [token for token in tokens if token not in
	stopword_list]
	return filtered_tokens

def remove_customized_strings(tokens):
	customized_list = [u'figure', u'table', u'also', u'use', u'eg', 'e.g.']
	filtered_tokens = [token for token in tokens if token not in
	customized_list]
	return filtered_tokens

def remove_numbers(text):
	return re.sub('[0-9]+', '', text)

def remove_special_characters(tokens):
	pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
	filtered_tokens = filter(None, [pattern.sub('', token) for token in
	tokens])
	return filtered_tokens

# tree = ET.parse('/Users/Revant/Desktop/CSpapers/ft/CIDR_17_008.cermxml')
# root = tree.getroot()
# abstract_text = root.find('front/article-meta/abstract/p').text

# expanded_abstract = expandContractions(abstract_text)
# lemmatized = lemmatize_text(expanded_abstract)
# tokenized_expanded_abstract = tokenize_text(lemmatized)

# stopwords = remove_stopwords(tokenized_expanded_abstract)
# special_characters = remove_special_characters(stopwords)

def extract_from_xml(path):
	tree = ET.parse(path)
	root1 = tree.getroot()
	root = tree.getroot()

	text = root1.find('front/article-meta/abstract/p').text
	for front in root.iter('front'):
		front.clear()

	for back in root.iter('back'):
		back.clear()

	# print(path)
	# print(text)
	if text is None:
		text = ""
	text = text + ' '.join(root.itertext())
	return text

def process_text(text):
	text = remove_numbers(text)
	text = expandContractions(text)
	text = lemmatize_text(text)
	text = tokenize_text(text)
	text = remove_stopwords(text)
	text = remove_customized_strings(text)
	text = remove_special_characters(text)

	return text


def bow_extractor(corpus, ngram_range):
    
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print df


def tfidf_transformer(bow_matrix):
    
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix
    
    

def tfidf_extractor(corpus, ngram_range=(1,1)):
    
    vectorizer = TfidfVectorizer(min_df=1, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def build_feature_matrix(documents, feature_type,
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):

    feature_type = feature_type.lower().strip()  
    
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                     ngram_range=ngram_range)
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    
    return vectorizer, feature_matrix


def compute_cosine_similarity(doc_features, corpus_features, top_n=3):
    # get document vectors
    doc_features = doc_features[0]
    # compute similarities
    similarity = np.dot(doc_features, 
                        corpus_features.T)
    similarity = similarity.toarray()[0]
    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 3))
                            for index in top_docs]
    return top_docs_with_score

# def main(path_list):

# 	# print "This is the name of the script: ", sys.argv[0]
# 	# print "Number of arguments: ", len(sys.argv)
# 	# print "The arguments are: " , str(sys.argv)
# 	array_strings = []
# 	text = extract_from_xml('/Users/Revant/Desktop/CSpapers/ft/CIDR_17_008.cermxml')
# 	text_list = process_text(text)
# 	# print(text_list)
# 	array_strings.append(' '.join(text_list))
# 	print(array_strings)

# 	# array_strings = []

# 	# for path in path_list:
# 	# 	text = extract_from_xml(path)
# 	# 	text_list = process_text(text)
# 	# 	array_strings.append(' '.join(text_list))

# 	# bow_vectorizer, bow_features = bow_extractor(array_strings, (1, 1))
# 	# # features = bow_features.todense()
# 	# # display_features(features, bow_vectorizer.get_feature_names())
	
# 	# transformer, tfidf_matrix = tfidf_transformer(bow_features.todense())

# 	# # wordcloud = WordCloud().generate(array_strings[-1])
# 	# # plt.imshow(wordcloud, interpolation='bilinear')
# 	# # plt.axis("off")

# 	# # wordcloud = WordCloud().generate(array_strings[-2])
# 	# # plt.imshow(wordcloud, interpolation='bilinear')
# 	# # plt.axis("off")

# 	# wordcloud = WordCloud(max_font_size=60).generate(array_strings[-1])
# 	# plt.figure()
# 	# plt.imshow(wordcloud, interpolation="bilinear")
# 	# plt.axis("off")

# 	# plt.show()



# def main(combined, individual):
	
# 	array_strings = []

# 	for each in sys.argv[1:]:
# 		each = '/Users/Revant/Desktop/CSpapers/ft/' + str(each)
# 		text = extract_from_xml(each)
# 		text_list = process_text(text)
# 		array_strings.append(' '.join(text_list))

# 	bow_vectorizer, bow_features = bow_extractor(array_strings, (1, 1))



# 	# features = bow_features.todense()
# 	# display_features(features, bow_vectorizer.get_feature_names())
	
# 	transformer, tfidf_matrix = tfidf_transformer(bow_features.todense())

# 	plt.figure(1)

# 	wordcloud = WordCloud().generate(array_strings[-1])
# 	plt.subplot(211)
# 	plt.imshow(wordcloud, interpolation='bilinear')
# 	plt.axis("off")

# 	wordcloud = WordCloud(max_font_size=60).generate(array_strings[-1])
# 	# plt.figure()
# 	plt.subplot(212)
# 	plt.imshow(wordcloud, interpolation="bilinear")
# 	plt.axis("off")

# 	plt.show()


def main(combined, individual, nameOfCombined, toy_corpus, query_docs):
	
	count = 1

	if combined:
		combined_text = ""

		for each_paper in combined:
			each_paper = '/Users/Revant/Desktop/CSpapers/ft/' + str(each_paper)
			combined_text += extract_from_xml(each_paper)

		combined_text_list = process_text(combined_text)
		plt.figure(count)
		wordcloud = WordCloud().generate(' '.join(combined_text_list))
		# plt.subplot(211)
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.savefig('/Users/Revant/Desktop/WordClouds/' + nameOfCombined)
		count += 1

	if individual:
		array_strings = []
		individual_names = []

		for each in individual:
			index = each.find('.')
			individual_names.append(each[:index] + ".png")


		for each_paper in individual:
			each_paper = '/Users/Revant/Desktop/CSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml(each_paper)
			individual_text_list = process_text(individual_text)
			array_strings.append(' '.join(individual_text_list))


		for i in range(len(array_strings)):
			plt.figure(count)
			wordcloud = WordCloud().generate(array_strings[i])
			# plt.subplot(211)
			plt.imshow(wordcloud, interpolation='bilinear')
			plt.axis("off")

			# wordcloud = WordCloud(max_font_size=60).generate(each)
			# plt.subplot(212)
			# plt.imshow(wordcloud, interpolation="bilinear")
			# plt.axis("off")

			plt.savefig("/Users/Revant/Desktop/" + "WordClouds/" + individual_names[i])
			count += 1

		bow_vectorizer, bow_features = bow_extractor(array_strings, (1, 1))
		# features = bow_features.todense()
		# display_features(features, bow_vectorizer.get_feature_names())
		transformer, tfidf_matrix = tfidf_transformer(bow_features.todense())



	#cosine similarity
	if toy_corpus:
		norm_corpus = []
		for each_paper in toy_corpus:
			each_paper = '/Users/Revant/Desktop/CSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml(each_paper)
			individual_text_list = process_text(individual_text)
			norm_corpus.append(' '.join(individual_text_list))

		tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_corpus,
                                                        feature_type='tfidf',
                                                        ngram_range=(1, 1), 
                                                        min_df=0.0, max_df=1.0)

	if query_docs:
		norm_query_docs = []
		for each_paper in query_docs:
			each_paper = '/Users/Revant/Desktop/CSpapers/ft/' + str(each_paper)
			individual_text = extract_from_xml(each_paper)
			individual_text_list = process_text(individual_text)
			norm_query_docs.append(' '.join(individual_text_list))

		query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)

	print 'Document Similarity Analysis using Cosine Similarity'
	print '='*60

	for index, doc in enumerate(query_docs):
	    
	    doc_tfidf = query_docs_tfidf[index]
	    top_similar_docs = compute_cosine_similarity(doc_tfidf,
	                                             tfidf_features,
	                                             top_n=2)
	    print 'Document',index+1 ,':', doc
	    print 'Top', len(top_similar_docs), 'similar docs:'
	    print '-'*40 
	    for doc_index, sim_score in top_similar_docs:
	        print 'Doc num: {} Similarity Score: {}\nDoc: {}'.format(doc_index+1,
	                                                                 sim_score,
	                                                                 toy_corpus[doc_index])  
	        print '-'*40       
	    print 



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--combined', nargs = '+', default = None)
	parser.add_argument('--nameOfCombined', default = None)
	parser.add_argument('--individual', nargs = '+', default = None)
	parser.add_argument('--toy_corpus', nargs = '+', default = None)
	parser.add_argument('--query_docs', nargs = '+', default = None)
	args = parser.parse_args()

	# if (not args.combined) and (not args.individual):
	# 	print("Sorry, too few arguments inputted.")
	# 	exit(1)

	# if (args.combined and not args.nameOfCombined):
	# 	print("Please provide name for the combined plot.")

	main(args.combined, args.individual, args.nameOfCombined, args.toy_corpus, args.query_docs)

