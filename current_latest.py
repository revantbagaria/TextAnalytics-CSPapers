import argparse, csv, glob
from os.path import expanduser


def main(dir_no):

	all_titles_vertical = []
	all_titles_horizontal = []

	with open("titles.txt", "r") as f:
		all_titles_horizontal.extend(f.readlines())

	for index, each in enumerate(all_titles_horizontal):
		if index != len(all_titles_horizontal)-1:
			all_titles_horizontal[index] = each[:len(each)-1]

	# home = expanduser("~")
	# filepath = home + "/Desktop/sessions/" + dir_no + ".txt"
	filepath = dir_no + ".txt"
	rows = []

	with open(filepath, "r") as f:
		rows.extend(f.readlines())

	for each in rows:
		current_row = []
		current_row.append(each.split()[1])
		# current_row[0] = current_row[0][:len(current_row[0])-1]
		current_row.extend([""] * len(all_titles_horizontal))
		all_titles_vertical.append(current_row)

	headings = ["Names"] + all_titles_horizontal

	filename = "similarity_half" + dir_no + ".csv"

	with open(filename, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(headings)
		writer.writerows(all_titles_vertical)



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-dn', '--dir_no', default = None)
	args = parser.parse_args()
	main(args.dir_no)