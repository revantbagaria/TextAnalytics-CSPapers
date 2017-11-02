from scipy.stats import itemfreq
import numpy as np

def vectorize_terms(terms):
	terms = [term.lower() for term in terms]
	terms = [np.array(list(term)) for term in terms]
	terms = [np.array([ord(char) for char in term]) for term in terms]
	return terms

def boc_term_vectors(word_list):
	word_list = [word.lower() for word in word_list]
	unique_chars = np.unique(np.hstack([list(word) for word in word_list]))
	word_list_term_counts = [{char: count for char, count in itemfreq(list(word))} for word in word_list]
	boc_vectors = [np.array([int(word_term_counts.get(char, 0)) for char in unique_chars])
							for word_term_counts in word_list_term_counts]
	return list(unique_chars), boc_vectors


# root = 'Believe'
# term1 = 'beleive'
# term2 = 'bargain'
# term3 = 'Elephant'

# terms = [root, term1, term2, term3]

# vec_root, vec_term1, vec_term2, vec_term3 = vectorize_terms(terms)

# # print(vec_root)
# # print(vec_term1)
# # print(vec_term2)
# # print(vec_term3)


# features, (boc_root, boc_term1, boc_term2, boc_term3) = boc_term_vectors(terms)

# # print(boc_root)
# # print(boc_term1)
# # print(boc_term2)
# # print(boc_term3)


# root_term = root
# root_vector = vec_root
# root_boc_vector = boc_root
# terms = [term1, term2, term3]
# vector_terms = [vec_term1, vec_term2, vec_term3]
# boc_vector_terms = [boc_term1, boc_term2, boc_term3]

# def cosine_distance(u, v):
# 	distance = 1.0 - (np.dot(u, v) / (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.square(v)))))
# 	return distance




# for term, boc_term in zip(terms, boc_vector_terms):
# 	print 'Analyzing similarity between root: {} and term: {}'.format(root_term, term)
# 	distance = round(cosine_distance(root_boc_vector, boc_term),2)
# 	similarity = 1 - distance
# 	print 'Cosine distance is {}'.format(distance)
# 	print 'Cosine similarity is {}'.format(similarity)
# 	print '-'*40
    

def compute_cosine_similarity(doc_features, corpus_features, top_n=3):
    # get document vectors
    doc_features = doc_features[0]
    # compute similarities
    similarity = np.dot(doc_features, 
                        corpus_features.T)
    similarity = similarity.toarray()[0]

    return similarity.tolist()

# def compute_cosine_similarity(doc_features, corpus_features, top_n=3):
#     # get document vectors
#     doc_features = doc_features[0]
#     # compute similarities
#     similarity = np.dot(doc_features, 
#                         corpus_features.T)
#     similarity = similarity.toarray()[0]

#     print("here goes the list: ")
#     print(similarity)

#     # get docs with highest similarity scores
#     top_docs = similarity.argsort()[::-1][:top_n]
#     top_docs_with_score = [(index, round(similarity[index], 3))
#                             for index in top_docs]
#     return top_docs_with_score
