import csv

titles = ["Name", "No of Papers", "Mean", "Median", "25th Quartile", "75th Quartile", "Std", "Max", "Min"]

with open("intraconf_stats_latest.csv", 'w') as f:
	writer = csv.writer(f)
	writer.writerow(titles)
