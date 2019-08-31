from market import *
from datetime import date

today = date.today()
fname = 'company_stats_' + str(today) + '.csv'
update_csv(csv_name=fname)