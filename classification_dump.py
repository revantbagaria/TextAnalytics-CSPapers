from sklearn.cross_validation import train_test_split
import findThePapers, generate_text_list, feature_extractor
from feature_extractor import bow_extractor, tfidf_extractor
from feature_extractor import averaged_word_vectorizer
from feature_extractor import tfidf_weighted_averaged_word_vectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import nltk, gensim, re, json
from sklearn import metrics
import pandas as pd
import numpy as np

d1 = {"Architecture": 0, "Benchmark": 1, "Cloud": 2, "Compilers": 3, "Concurrency": 4, "Data": 5, "DB": 6, "Energy": 7, "GPGPU": 8, "HPC": 9, "Network": 10, "OS": 11, "Security": 12, "Storage": 13, "VM": 14}
d2 = {0: "Architecture", 1: "Benchmark", 2: "Cloud", 3: "Compilers", 4: "Concurrency", 5: "Data", 6: "DB", 7: "Energy", 8: "GPGPU", 9: "HPC", 10: "Network", 11: "OS", 12: "Security", 13: "Storage", 14: "VM"}


def get_metrics(true_labels, predicted_labels):
    
    print 'Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        2)
    print 'Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2)
    print 'Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2)
    print 'F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2)
                        

def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    # evaluate model prediction performance   
    get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)
    return predictions  


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
	train_papers, test_papers, train_labels, test_labels = prepare_datasets(papers,
                                                                        vectors,
                                                                        test_data_proportion=0.3)
	norm_train_corpus = generate_text_list.generate_text_list(train_papers)
	norm_test_corpus = generate_text_list.generate_text_list(test_papers)


	# bag of words features
	bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)  
	bow_test_features = bow_vectorizer.transform(norm_test_corpus) 

	# tfidf features
	tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)  
	tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)    


	# tokenize documents
	tokenized_train = [nltk.word_tokenize(text)
	                   for text in norm_train_corpus]
	tokenized_test = [nltk.word_tokenize(text)
	                   for text in norm_test_corpus]  
	# build word2vec model                   
	model = gensim.models.Word2Vec(tokenized_train,
	                               size=500,
	                               window=100,
	                               min_count=30,
	                               sample=1e-3)                  
	                   
	# averaged word vector features
	avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train,
	                                                 model=model,
	                                                 num_features=500)                   
	avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test,
	                                                model=model,
	                                                num_features=500)                                                 
	                   


	# tfidf weighted averaged word vector features
	vocab = tfidf_vectorizer.vocabulary_
	tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train, 
	                                                                  tfidf_vectors=tfidf_train_features, 
	                                                                  tfidf_vocabulary=vocab, 
	                                                                  model=model, 
	                                                                  num_features=500)
	tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test, 
	                                                                 tfidf_vectors=tfidf_test_features, 
	                                                                 tfidf_vocabulary=vocab, 
	                                                                 model=model, 
	                                                                 num_features=500)

	mnb = MultinomialNB()
	svm = SGDClassifier(loss='hinge', n_iter=100)

	# Multinomial Naive Bayes with bag of words features
	mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
	                                           train_features=bow_train_features,
	                                           train_labels=train_labels,
	                                           test_features=bow_test_features,
	                                           test_labels=test_labels)

	# Support Vector Machine with bag of words features
	svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
	                                           train_features=bow_train_features,
	                                           train_labels=train_labels,
	                                           test_features=bow_test_features,
	                                           test_labels=test_labels)
	                                           
	# Multinomial Naive Bayes with tfidf features                                           
	mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
	                                           train_features=tfidf_train_features,
	                                           train_labels=train_labels,
	                                           test_features=tfidf_test_features,
	                                           test_labels=test_labels)

	# Support Vector Machine with tfidf features
	svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
	                                           train_features=tfidf_train_features,
	                                           train_labels=train_labels,
	                                           test_features=tfidf_test_features,
	                                           test_labels=test_labels)

	# Support Vector Machine with averaged word vector features
	svm_avgwv_predictions = train_predict_evaluate_model(classifier=svm,
	                                           train_features=avg_wv_train_features,
	                                           train_labels=train_labels,
	                                           test_features=avg_wv_test_features,
	                                           test_labels=test_labels)

	# Support Vector Machine with tfidf weighted averaged word vector features
	svm_tfidfwv_predictions = train_predict_evaluate_model(classifier=svm,
	                                           train_features=tfidf_wv_train_features,
	                                           train_labels=train_labels,
	                                           test_features=tfidf_wv_test_features,
	                                           test_labels=test_labels)

	 
	cm = metrics.confusion_matrix(test_labels, svm_tfidf_predictions)
	pd.DataFrame(cm, index=range(0,20), columns=range(0,20))  

	# class_names = dataset.target_names
	# print class_names[0], '->', class_names[15]
	# print class_names[18], '->', class_names[16]  
	# print class_names[19], '->', class_names[15] 