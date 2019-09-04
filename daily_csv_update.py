from market import *
from datetime import date

today = date.today()
fname = 'company_stats_' + str(today) + '.csv'
update_csv(csv_name=fname)

with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_statistics.csv', 'w', newline='') as dest:   
    writer = csv.writer(dest)
    with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_stats_' + str(today) + '.csv', 'r', newline='') as source:
        reader = csv.reader(source)
        for row in reader:
            writer.writerow(row) 

