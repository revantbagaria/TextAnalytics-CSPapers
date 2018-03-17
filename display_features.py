import pandas as pd
import csv


def edit_csv_file(filename):
	new_rows = [] 

	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			new_row = []
			for each in row:
				try:
					float(each)
					if abs(float(each) - 1.00) <= 0.01:
						new_row.append("")
					else:
						new_row.append(each)
				except ValueError:
					new_row.append(each)
			new_rows.append(new_row)

	with open(filename, 'wb') as f:
		writer = csv.writer(f)
		writer.writerows(new_rows)


def display_features(features, feature_names, index_names=None):
    df = pd.DataFrame(data=features, index=index_names,
                      columns=feature_names)
    print df
    # filename = "similarity_matrix_combined.csv"
    # df.to_csv(filename)
    # edit_csv_file(filename)