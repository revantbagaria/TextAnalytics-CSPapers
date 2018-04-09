from os.path import expanduser


home = expanduser("~")
starting_col = 2
file_count = 2
all_titles = []
index = 0

with open("titles.txt", "r") as f:
	all_titles.extend(f.readlines())

for i in range(20):
	filepath = home + "/Desktop/sessions/" + str(file_count) + ".txt"
	with open(filepath, "w") as fp:
		for j in range(15):
			line = str(starting_col) + " " + all_titles[index]
			fp.write(line)
			index += 1
			starting_col += 1
	file_count += 1

for i in range(40):
	filepath = home + "/Desktop/sessions/" + str(file_count) + ".txt"
	with open(filepath, "w") as fp:
		for j in range(20):
			line = str(starting_col) + " " + all_titles[index]
			fp.write(line)
			index += 1
			starting_col += 1
	file_count += 1


for i in range(20):
	filepath = home + "/Desktop/sessions/" + str(file_count) + ".txt"
	with open(filepath, "w") as fp:
		for j in range(34):

			if index == 1769:
				break

			line = str(starting_col) + " " + all_titles[index]
			fp.write(line)
			index += 1
			starting_col += 1

	file_count += 1
