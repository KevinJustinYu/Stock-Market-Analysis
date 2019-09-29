from market import *
from datetime import date, timedelta

yesterday = date.today() #- timedelta(1)
fname = 'company_stats_' + str(yesterday) + '.csv'
update_csv(csv_name=fname)

with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_statistics.csv', 'w', newline='') as dest:   
    writer = csv.writer(dest)
    with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_stats_' + str(yesterday) + '.csv', 'r', newline='') as source:
        reader = csv.reader(source)
        for row in reader:
            writer.writerow(row) 

