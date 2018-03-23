from contraction_mapping import contraction_dict
import re, nltk, string, sys
from nltk.stem import WordNetLemmatizer
from pattern.en import tag
from nltk.corpus import wordnet as wn


def tokenize_text(text):
	text = text.lower() #important step remember we are converting everything to lower case
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

def lemmatize_text(text):
	wnl = WordNetLemmatizer()
	pos_tagged_text = pos_tag_text(text)
	lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
	lemmatized_text = ' '.join(lemmatized_tokens)
	return lemmatized_text

def remove_stopwords(tokens):
	stopword_list = nltk.corpus.stopwords.words('english')
	filtered_tokens = [token for token in tokens if token not in stopword_list]
	return filtered_tokens

def remove_customized_strings(tokens):
	customized_list = [u'figure', u'table', u'also', u'use', u'eg', 'e.g.']
	filtered_tokens = [token for token in tokens if token not in customized_list]
	return filtered_tokens

def remove_numbers(text):
	return re.sub('[0-9]+', '', text)

def remove_special_characters(tokens):
	pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
	filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
	return filtered_tokens

# def mapping(text):
	

def process_text(text):
	text = remove_numbers(text)
	text = expandContractions(text)
	text = lemmatize_text(text)
	text = tokenize_text(text)
	text = remove_stopwords(text)
	text = remove_customized_strings(text)
	text = remove_special_characters(text)

	return text