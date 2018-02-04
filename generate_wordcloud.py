from wordcloud import WordCloud
import matplotlib.pyplot as plt
import globalVariables, findThePapers, generate_text_list


def wordcloud(text, name_of_wordcloud):
	plt.figure(globalVariables.countGraphs)
	globalVariables.countGraphs += 1
	wordcloud = WordCloud().generate(text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig('/Users/Revant/Desktop/ThesisFiles/WordClouds/' + name_of_wordcloud)

def generate_wordcloud(wordcloud_combfiles, wordcloud_indfiles):

	if (not wordcloud_combfiles) and (not wordcloud_indfiles):
		print("Please enter files for which WordCloud need to be generated.")
		exit(1)

	if wordcloud_combfiles:
		wordcloud_combfiles_extended = findThePapers.findThePapers(wordcloud_combfiles)
		text_list = generate_text_list.generate_text_list(wordcloud_combfiles_extended)
		for index, each in enumerate(text_list):
			wordcloud(each, "combined" + str(index+1) + ".png")

	if wordcloud_indfiles:
		wordcloud_indfiles_extended = findThePapers.findThePapers(wordcloud_indfiles)
		text_list = generate_text_list.generate_text_list(wordcloud_indfiles_extended)
		for index, each in enumerate(text_list):
			wordcloud(each, wordcloud_indfiles[index] + ".png")