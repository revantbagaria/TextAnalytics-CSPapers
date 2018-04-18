import csv
import glob, pickle
from os.path import expanduser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


home = expanduser("~")
path = home + '/Desktop/ThesisFiles/IntraConf/*.csv'
result = []

for each in glob.glob(path):
	result.append(each)


similarities_all = []

for each in result:
	similarities = []
	index = each.rfind('/')
	name = each[index+1:]
	with open(each, 'r') as f:
		reader = csv.reader(f)
		titles = next(reader)
		for i, row in enumerate(reader):
			for j, each in enumerate(row):
				try:
					float(each)
					t = float(each)
					similarities.append(t)
				except ValueError:
					continue

	similarities = np.array(similarities)
	similarities_all.append((similarities, name))

similarities_all.append((pickle.load(open("all_to_all_similarities.txt", "r")), "ALL_to_ALL.csv"))
similarities_all.sort(reverse=True, key=lambda a: np.mean(a[0]))

labels, data = [], []

# labels.append("ALL_to_ALL")
# data.append(pickle.load(open("all_to_all_similarities.txt", "r")))

for each in similarities_all:
	name = each[1]
	name = name[:(len(name)-4)]
	labels.append(name)
	data.append(each[0])

# data = np.log10(data)
plt.style.use('ggplot')
plt.boxplot(data, vert=False, showmeans=True)
plt.yticks(np.arange(len(labels))+1, labels)
plt.legend()
plt.title("Intra-Conference & All-to-All Similarities (sorted by mean)")
plt.show()
