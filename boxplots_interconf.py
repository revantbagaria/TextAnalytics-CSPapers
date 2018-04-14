import csv
import matplotlib.pyplot as plt
from os.path import expanduser
import pandas as pd
import numpy as np
from scipy.stats import norm

home = expanduser("~")
path = home + "/Desktop/ThesisFiles/InterConf/similarity_combined_full.csv"

similarities_all = []

with open(path, 'r') as f:
	reader = csv.reader(f)
	titles = next(reader)
	titles = titles[1:]

	for i in range(44):
		row = next(reader)
		row = row[1:]
		similarities = []
		for j in range(len(row)):
			try:
				float(row[j])
				similarities.append(float(row[j]))
			except ValueError:
				continue
		similarities = np.array(similarities)
		similarities_all.append((similarities, titles[i]))

similarities_all.sort(reverse=True, key=lambda a: np.std(a[0]))
labels, data = [], []

for each in similarities_all:
	labels.append(each[1])
	data.append(each[0])

plt.style.use('ggplot')
plt.boxplot(data, vert=False, showmeans=True)
plt.yticks(np.arange(len(labels))+1, labels)
plt.legend()
plt.title("Inter-Conference Similarities (sorted by std)")


# all_similarities_values = []
# for each1 in data:
# 	for each2 in each1:
# 		all_similarities_values.append(each2)

# plt.style.use('ggplot')
# df = pd.DataFrame()
# df['similarities'] = all_similarities_values
# df['index_col'] = range(1, len(df)+1)
# df.plot(kind='density', title = 'Similarity Density', logx=True)

# all_similarities_values = np.array(all_similarities_values)
# plt.plot(all_similarities_values, norm.cdf(all_similarities_values))
plt.show()