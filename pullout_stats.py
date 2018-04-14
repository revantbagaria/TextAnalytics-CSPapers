import csv
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def pullout_stats(filename):
	similarities = []
	no_papers = 0
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		titles = next(reader)
		no_papers += len(titles)
		for i, row in enumerate(reader):
			for j, each in enumerate(row):
				try:
					float(each)
					t = (float(each), (titles[i+1], titles[j]))
					similarities.append(t)
				except ValueError:
					continue
	
	similarities.sort(key = lambda a: a[0], reverse=True)
	
	print("Following the highest 10 similarity pairs:")
	for i in range(0, 20, 2):
		print(similarities[i])

	print("")

	print("Following the lowest 10 similarity pairs:")
	for i in range(len(similarities)-20, len(similarities), 2):
		print(similarities[i])

	print("")

	extract_similarities = []

	for each in similarities:
		extract_similarities.append(each[0])

	extract_similarities = np.array(extract_similarities)

	print("Mean: %f" % np.mean(extract_similarities))
	print("Median: %f" % np.median(extract_similarities))
	print("25th percentile: %f" % np.percentile(extract_similarities, 25))
	print("50th percentile: %f" % np.percentile(extract_similarities, 50))
	print("75th percentile: %f" % np.percentile(extract_similarities, 75))
	print("Max: %f" % np.max(extract_similarities))
	print("Min: %f" % np.min(extract_similarities))
	print("Standard Deviation: %f" % np.std(extract_similarities))	

	# name = filename[:(len(filename)-4)]
	
	# row = []
	# row.append(name)
	# row.append(str(no_papers))
	# row.append(str(np.mean(extract_similarities)))
	# row.append(str(np.median(extract_similarities)))
	# row.append(str(np.percentile(extract_similarities, 25)))
	# row.append(str(np.percentile(extract_similarities, 75)))
	# row.append(str(np.std(extract_similarities)))
	# row.append(str(np.max(extract_similarities)))
	
	# row.append(str(np.min(extract_similarities)))

	# with open("intraconf_stats_latest.csv", "a") as f:
	#     writer = csv.writer(f)
	#     writer.writerow(row)


	# df = pd.DataFrame()
	# df['similarities'] = extract_similarities
	# df['index_col'] = range(1, len(df)+1)

	# plt.style.use('ggplot')
	# print(pd.DataFrame(extract_similarities).count)
	# print(pd.DataFrame(extract_similarities).shape)
	# ax = df.plot(kind='density', title = 'Similarity Density', logx = True)
	# df.plot(kind='scatter', title= 'Similarity for individual papers', x='index_col', y='similarities', alpha = 0.03)
	# ax = df.plot.hexbin(gridsize =25, x='index_col', y='similarities', yscale = 'log', title= 'Similarity for individual papers')
	# ax = df.plot.hexbin(gridsize =25, x='index_col', y='similarities', title= 'Similarity for individual papers')
	# plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', default = None)
	args = parser.parse_args()
	pullout_stats(args.filename)

