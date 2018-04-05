import csv

titles = []
row = []

conf_names = "ASPLOS ATC CCGRID CCS CIDR CoNEXT EuroPar EuroSys FAST HCW HotCloud HotI HotOS HotStorage HPCA HPDC ICAC ICPE IMC ISC ISCA ISPASS KDD MASCOTS MICRO Middleware MobiCom NDSS PLDI PODC PODS PPoPP SC SIGCOMM SIGIR SIGMETRICS SIGMOD SLE SOCC SOSP SP SPAA SYSTOR VEE".split()

for name in conf_names:
	titles.append(name)

for name in conf_names:
	curr_row = []
	curr_row.append(name)
	curr_row = curr_row + ([""] * len(titles))
	row.append(curr_row)

titles = ["Names"] + titles

with open('similarity_combined_half.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(titles)
	writer.writerows(row)

with open('similarity_combined_full.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(titles)
	writer.writerows(row)