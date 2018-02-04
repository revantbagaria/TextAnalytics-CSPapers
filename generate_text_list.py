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