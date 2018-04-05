import argparse
from os.path import expanduser
import glob, csv

def main(conf):
	titles = []
	row = []
	home = expanduser("~")
	path = home + '/Desktop/ThesisCSpapers/' + conf + '.cermxml'

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

	filename = conf[:(len(conf)-2)] + ".csv"

	with open(filename, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(titles)
		writer.writerows(row)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-conf', '--conf', default = None)
	args = parser.parse_args()
	main(args.conf)