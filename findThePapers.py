import glob
from os.path import expanduser


def delete_empty_ones(result):
	empty_files = ["CCGRID_17_012", "ISPASS_17_007", "Middleware_17_019", "NSDI_17_009", "NSDI_17_037", "NSDI_17_034", "NSDI_17_040", "NSDI_17_042", "NSDI_17_036", "NSDI_17_038", "NSDI_17_035", "NSDI_17_032", "NSDI_17_039", "NSDI_17_041", "NSDI_17_030", "NSDI_17_031", "NSDI_17_024", "NSDI_17_033", "ICPE_17_022"]
	home = expanduser("~")

	for each in empty_files:
		each = home + '/Desktop/ThesisCSpapers/' + each + '.cermxml'
		if each in result:
			result.remove(each)
			
	return result

def findCompletePath(filename):
	result = []
	if filename[-1] == '*':
			home = expanduser("~")
			path = home + '/Desktop/ThesisCSpapers/' + str(filename) + '.cermxml'
			for name in glob.glob(path):
				result.append(name)
	else:
		home = expanduser("~")
		result.append(home + '/Desktop/ThesisCSpapers/' + str(filename) + '.cermxml')

	result = delete_empty_ones(result) #this is where deletion is happening
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