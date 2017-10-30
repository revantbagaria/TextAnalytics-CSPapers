from wordcloud import WordCloud
import matplotlib.pyplot as plt


def wordcloud(text, name_of_wordcloud, count):
	plt.figure(count)
	wordcloud = WordCloud().generate(' '.join(text))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig('/Users/Revant/Desktop/ThesisFiles/WordClouds/' + name_of_wordcloud)