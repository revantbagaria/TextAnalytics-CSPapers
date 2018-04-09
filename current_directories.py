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


with open("titles.txt", "wb") as f:
	for i, each in enumerate(titles):
		if i != (len(titles)-1):
			each += "\n"
		f.write(each)
