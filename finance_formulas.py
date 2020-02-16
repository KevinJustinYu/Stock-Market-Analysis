import numpy as np

# ---------- Function Declarations ----------
def net_working_capital(current_assets, current_liabilities):
	return current_assets - current_liabilities


def return_on_investment(profit, investment_amount):
	return profit / investment_amount


def present_value_of_future_value(future_value, discount_rate, years):
	return future_value / (1 + discount_rate)**years


def present_value_of_stream(cash_flow_stream, discount_rate):
	n = len(cash_flow_stream)
	pv = 0

	# Bring each future cash flow to present value and sum
	for i in range(n):
		pv += present_value_of_future_value(cash_flow_stream[i], discount_rate, i)
	return pv