
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split



def get_data():
    data = fetch_20newsgroups(subset='all',
                              shuffle=True,
                              remove=('headers', 'footers', 'quotes'))
    return data

def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels

def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, 
                                                        test_size=0.33, random_state=42)
    return train_X, test_X, train_Y, test_Y

dataset = get_data()

corpus, labels = dataset.data, dataset.target
corpus, labels = corpus[:9], [[1,0,1], [0,0,1], [1,1,0], [0,1,0], [1,1,1], [1,0,0], [0,1,0], [1,1,1], [1,0,0]]
corpus, labels = remove_empty_docs(corpus, labels)




# print("labels :", labels)
# print("corpus: ", corpus)

# print(dataset)


# print(dataset.keys())
# print(dataset.target_names)
# print(dataset.target)
# print 'Sample document:', corpus[1]
# print 'Class label:',labels[1]
# print 'Actual class label:', dataset.target_names[labels[1]]


# def prepare_datasets(corpus, labels, test_data_proportion=0.3):
#     train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels, 
#                                                         test_size=0.33, random_state=42)
#     return train_X, test_X, train_Y, test_Y



# corpus = [
# "My brother is in the market for a high-performance video card that supports VESA local bus with 1-2MB RAM.  Does anyone have suggestions/ideas on",
# "I am  bit puzzled too and a bit relieved. However, I am going to put an end to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they are killing those Devils worse than I thought. Jagr just showed you why he is much better than his regular season stats. He is also a lot fo fun to watch in the playoffs. Bowman should let JAg", "Search Turkish planes? You don't know what you are talking about. i Turkey's government has announced that it's giving weapons  <-----------i to Azerbadjan since Arme"]

# labels = [[1,0,1], [0,0,1], [0,1,0]]

# train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus, labels, test_data_proportion=0.3)