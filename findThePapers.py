import glob

def findCompletePath(filename):
	result = []
	if filename[-1] == '*':
			path = '~/Desktop/ThesisCSpapers/' + str(filename) + '.cermxml'
			for name in glob.glob(path):
				result.append(name)
	else:
		result.append('~/Desktop/ThesisCSpapers/' + str(filename) + '.cermxml')
	return result

def findThePapers(files):
	result = []
	if not isinstance(files[0], list):
		for each in files:
			result.extend(findCompletePath(each))
	else:
		for each in files:
			res = []
			for f in each:
				res.extend(findCompletePath(f))
			result.append(res)
	return result