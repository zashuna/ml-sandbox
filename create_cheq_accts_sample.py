import csv

cheq_acct_file = '/STORAGE/cpb/data/CustomersInBothSecondaryAndSample.txt'
secondary_csv_file = open('/STORAGE/cpb/data/cheqSecondarySummary-Jahir-v8.csv', 'r', newline='')
secondary_csv_file_sample = open('/STORAGE/cpb/data/cheqSecondarySummary-Jahir-v8-sample.csv', 'w', newline='')

sample_writer = csv.writer(secondary_csv_file_sample, delimiter=',')
secondary_reader = csv.reader(secondary_csv_file)

cheq_accts = set()
with open(cheq_acct_file, 'r') as f:
    for line in f:
        cheq_accts.add(line.strip())

for row in secondary_reader:
    if row[0] in cheq_accts:
        sample_writer.writerow(row)

secondary_csv_file_sample.close()
secondary_csv_file.close()