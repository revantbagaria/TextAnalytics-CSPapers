import text_normalization, extract_from_xml

def generate_text_list(files_list):
	
	result = []

	for f in files_list:
		text = ""
		if isinstance(f, list):
			for each in f:
				text += " " + extract_from_xml.extract_from_xml(each)
		else:
			text = extract_from_xml.extract_from_xml(f)
		text = text_normalization.process_text(text)
		result.append(' '.join(text))

	return result


def generate_titles(files_list):

	result = []

	for f in files_list:
		if isinstance(f, list):
			title = f[0]
			index1 = f[0].rfind('/')
			index2 = f[0].find('_')
			result.append(f[0][index+1 : index2])
		else:
			index1 = f.rfind('/')
			index2 = f.find('.')
			result.append(f[index+1 : index2])
	return result
