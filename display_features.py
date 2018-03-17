import pandas as pd
import csv


def edit_csv_file(filename):
	new_rows = [] # a holder for our modified rows when we make them

	with open(filename, 'rb') as f:
	    reader = csv.reader(f) # pass the file to our csv reader
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
	    # Overwrite the old file with the modified rows
	    writer = csv.writer(f)
	    writer.writerows(new_rows)


def display_features(features, feature_names, index_names=None):
    df = pd.DataFrame(data=features, index=index_names,
                      columns=feature_names)
    # print df
    filename = "similarity_matrix_individual.csv"
    df.to_csv(filename)
    edit_csv_file(filename)







# import csv

# new_rows = [] # a holder for our modified rows when we make them
# changes = {   # a dictionary of changes to make, find 'key' substitue with 'value'
#     '1 dozen' : '12', # I assume both 'key' and 'value' are strings
#     }

# with open('test.csv', 'rb') as f:
#     reader = csv.reader(f) # pass the file to our csv reader
#     for row in reader:     # iterate over the rows in the file
#         new_row = row      # at first, just copy the row
#         for key, value in changes.items(): # iterate over 'changes' dictionary
#             new_row = [ x.replace(key, value) for x in new_row ] # make the substitutions
#         new_rows.append(new_row) # add the modified rows

# with open('test.csv', 'wb') as f:
#     # Overwrite the old file with the modified rows
#     writer = csv.writer(f)
#     writer.writerows(new_rows)