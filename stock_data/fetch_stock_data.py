from pandas_datareader import data
import pandas as pd
from time import sleep

tickers = ['AAPL','ABBV','ABT','ACN','AGN','AIG','ALL','AMGN','AMZN','AXP',
           'BA','BAC','BIIB','BK','BLK','BMY','C','CAT','CELG',
           'CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR',
           'DIS',
           'DUK','EMR','EXC','F','FB','FDX','FOX','GD',
           'GE','GILD','GM','GOOG','GS','HAL','HD','HON','IBM','INTC',
           'JNJ','JPM','KHC','KMI','KO','LLY','LMT','LOW','MA','MCD',
           'MDLZ','MDT','MET','MMM','MO',
           'MRK','MS','MSFT','NEE',
           'NKE','ORCL','OXY',
           'PEP','PFE','PG','PM','PYPL','QCOM',
           'RTN','SBUX','SLB','SO','SPG','T','TGT','TXN','UNH',
           'UNP','UPS','USB','UTX','V','VZ','WBA','WFC','WMT','XOM']

# BRK, DOW, MON, PCLN excluded
# TWX does not exist in 2018
#tickers = ['AAPL','ABBV']

start_date = '2016-01-04'
end_date = '2018-12-30'

df = pd.DataFrame()

for tick in tickers:
    print(tick)
    df = pd.concat([df,
                    data.DataReader(tick, 'yahoo', start_date, end_date)['Close']],
                   axis=1, sort=False)

df.set_axis(tickers, axis=1, inplace=True)
    
df.to_pickle('stock_dataframe.pkl')
