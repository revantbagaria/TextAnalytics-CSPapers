def generate_titles(files_list):

	result = []

	for f in files_list:
		if isinstance(f, list):
			title = f[0]
			index1 = f[0].rfind('/')
			index2 = f[0].find('_')
			result.append(f[0][index1+1 : index2])
		else:
			index1 = f.rfind('/')
			index2 = f.find('.')
			result.append(f[index1+1 : index2])
	return result