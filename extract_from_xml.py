import xml.etree.ElementTree as ET


def extract_from_xml(path):
	tree = ET.parse(path)
	root1 = tree.getroot()
	root = tree.getroot()

	text = root1.find('front/article-meta/abstract/p').text
	for front in root.iter('front'):
		front.clear()

	for back in root.iter('back'):
		back.clear()

	# print(path)
	# print(text)
	if text is None:
		text = ""
	text = text + ' '.join(root.itertext())
	return text