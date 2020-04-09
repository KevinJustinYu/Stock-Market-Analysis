import numpy as np

# ---------- Function Declarations ----------
def convert_to_real_return(nominal_return, inflation):
    ''' Computes the real return (the increase in actual buying power),
        by taking inflation into account. 
        Pass in inflation as a single value for one year, or a list of values 
        for multiple years. '''
    # For multiple years, calculate inflation rate
    if isinstance(inflation, list):
        inflation = np.prod(inflation) 
    return (nominal_return - inflation) / (1 + inflation)


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