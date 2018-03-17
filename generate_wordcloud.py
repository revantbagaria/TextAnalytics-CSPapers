import matplotlib
matplotlib.use('Agg')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import globalVariables, findThePapers, generate_text_list
from generate_titles import generate_titles
from os.path import expanduser


def wordcloud(text, name_of_wordcloud, combined, individual):
	plt.figure(globalVariables.countGraphs)
	globalVariables.countGraphs += 1
	wordcloud = WordCloud().generate(text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	home = expanduser("~")
	
	if combined:
		path = home + '/Desktop/ThesisFiles/WordClouds/Combined/' + name_of_wordcloud
	else:
		path = home + '/Desktop/ThesisFiles/WordClouds/Individual/' + name_of_wordcloud

	plt.savefig(path)

	# plt.savefig('/Users/Revant/Desktop/ThesisFiles/WordClouds/' + name_of_wordcloud)

def generate_wordcloud(wordcloud_combfiles, wordcloud_indfiles):

	if (not wordcloud_combfiles) and (not wordcloud_indfiles):
		print("Please enter files for which WordCloud need to be generated.")
		exit(1)

	if wordcloud_combfiles:
		wordcloud_combfiles_extended = findThePapers.findThePapers(wordcloud_combfiles)
		titles = generate_titles(wordcloud_combfiles_extended)
		text_list = generate_text_list.generate_text_list(wordcloud_combfiles_extended)
		for index, each in enumerate(text_list):
			name_of_wordcloud = "wordcloud_" + titles[index] + ".png"
			print(name_of_wordcloud)
			wordcloud(each, name_of_wordcloud, True, False)

	if wordcloud_indfiles:
		wordcloud_indfiles_extended = findThePapers.findThePapers(wordcloud_indfiles)
		titles = generate_titles(wordcloud_indfiles_extended)
		text_list = generate_text_list.generate_text_list(wordcloud_indfiles_extended)
		for index, each in enumerate(text_list):
			name_of_wordcloud = "wordcloud_" + titles[index] + ".png"
			print(name_of_wordcloud)
			wordcloud(each, name_of_wordcloud, False, True)