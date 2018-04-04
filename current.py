from os.path import expanduser
import glob, csv

titles = []
row = []
home = expanduser("~")
path = home + '/Desktop/ThesisCSpapers/' + '*' + '.cermxml'

for name in glob.glob(path):
	index1 = name.rfind('/')
	index2 = name.rfind('.')
	titles.append(name[(index1+1):index2])

for name in glob.glob(path):
	index1 = name.rfind('/')
	index2 = name.rfind('.')
	curr_row = []
	curr_row.append(name[(index1+1):index2])
	curr_row = curr_row + ([""] * len(titles))
	row.append(curr_row)
	

titles = ["Names"] + titles

with open('similarity_half.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(titles)
	writer.writerows(row)

with open('similarity_full.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(titles)
	writer.writerows(row)
