import findThePapers, generate_text_list, feature_extractor
from feature_extractor import bow_extractor, tfidf_extractor
from feature_extractor import averaged_word_vectorizer
from feature_extractor import tfidf_weighted_averaged_word_vectorizer
import nltk, gensim, re, json
from sklearn import metrics
import pandas as pd
import numpy as np
import scipy.sparse as sp


from sklearn.cross_validation import train_test_split
from skmultilearn.adapt import MLkNN
from skmultilearn.neurofuzzy import MLARAM


from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB



d1 = {"Architecture": 0, "Benchmark": 1, "Cloud": 2, "Compilers": 3, "Concurrency": 4, "Data": 5, "DB": 6, "Energy": 7, "GPGPU": 8, "HPC": 9, "Network": 10, "OS": 11, "Security": 12, "Storage": 13, "VM": 14}
d2 = {0: "Architecture", 1: "Benchmark", 2: "Cloud", 3: "Compilers", 4: "Concurrency", 5: "Data", 6: "DB", 7: "Energy", 8: "GPGPU", 9: "HPC", 10: "Network", 11: "OS", 12: "Security", 13: "Storage", 14: "VM"}



def adapt_knn(train_X, test_X, train_Y, test_Y):
	train_X, test_X  = sp.csr_matrix(train_X), sp.csr_matrix(test_X)
	train_Y, test_Y = sp.csr_matrix(train_Y), sp.csr_matrix(test_Y)
	classifier = MLkNN(k=3)
	classifier.fit(train_X, train_Y)
	predictions = classifier.predict(test_X)

def neurofuzzy_mlaram(train_X, test_X, train_Y, test_Y):
	train_X, test_X = np.matrix(train_X), np.matrix(test_X)
	train_Y, test_Y = np.array(train_Y), np.array(test_Y)
	classifier = MLARAM(vigilance=0.9, threshold=0.02, neurons=[])
	classifier.fit(train_X, train_Y)
	predictions = classifier.predict(test_X)
	print(type(predictions))
	print(predictions[0])


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, 
                                                        test_size=0.33, random_state=42)
    return train_X, test_X, train_Y, test_Y

def get_labels_files(classification_labels):
	result = []
	for each in classification_labels:
		path = '/Users/Revant/Desktop/ThesisCSpapers/' + str(each) + '.json'
		result.append(path)
	return result

def get_labels(label_files):
	papers, labels = [], []

	for f in label_files:
		with open(f) as json_file:
			json_data = json.load(json_file)
			for each in json_data["papers"]:
				papers.append(each["key"])
				labels.append(each["topics"])

	return papers, labels

def label_to_vector(labels):
	result = []
	for each in labels:
		vector = [0] * 15
		for i in each:
			vector[d1[i]] = 1
		result.append(vector)
	return result

def generate_classification(classification_files, classification_labels):
	if not classification_labels or not classification_files:
		print("Please enter both, classification_files and classification_labels.")
		exit(1)

	label_files = get_labels_files(classification_labels)
	papers, labels = get_labels(label_files)
	vectors = label_to_vector(labels)
	papers = findThePapers.findThePapers(papers)
	corpus = generate_text_list.generate_text_list(papers)

	train_X, test_X, train_Y, test_Y = prepare_datasets(corpus,
                                                        vectors,
                                                        test_data_proportion=0.3)

	# tfidf features
	tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_X)  
	tfidf_test_features = tfidf_vectorizer.transform(test_X) 

	# adapt_knn(tfidf_train_features, tfidf_test_features, train_Y, test_Y)

	# issues with the following:
	# neurofuzzy_mlaram(tfidf_train_features, tfidf_test_features, train_Y, test_Y)




	# test_Y = sp.csr_matrix(test_Y)
	# print(test_Y[:3])
	



	train_X, test_X  = sp.csr_matrix(tfidf_train_features), sp.csr_matrix(tfidf_test_features)
	train_Y, test_Y = sp.csr_matrix(train_Y), sp.csr_matrix(test_Y)

	classifier = LabelPowerset(GaussianNB())
	# train
	classifier.fit(train_X, train_Y)

	# predict
	predictions = classifier.predict(test_X)