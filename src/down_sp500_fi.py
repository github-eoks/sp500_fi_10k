# Ref: https://analyzingalpha.com/yfinance-python

import yfinance as yf
import pandas as pd
import time
import datetime





list_sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = list_sp['Symbol'].tolist()
tickers = tickers[:2]



# ticker1_yf = yf.Ticker(ticker_list[2])
# ticker1_yf = yf.Ticker('tsla')
#
# ticker1_yf.financials
# ticker1_yf.cashflow
# ticker1_yf.earnings
# ticker1_yf.balancesheet
# ticker1_yf.news

now = datetime.datetime.now()
print('Start time: ', now)
print('Collecting...')


######################
tickers = [yf.Ticker(ticker) for ticker in tickers]


dfs = []  # list for each ticker's dataframe
for ticker in tickers:
    i = tickers.index(ticker)
    if i% 1 == 0:
        print('Processing...', i, "/", len(tickers), tickers[i])
    # get each financial statement
    pnl = ticker.financials
    bs = ticker.balancesheet
    cf = ticker.cashflow

    # concatenate into one dataframe
    fs = pd.concat([pnl, bs, cf])

    # make dataframe format nicer
    # Swap dates and columns
    data = fs.T
    # reset index (date) into a column
    data = data.reset_index()
    # Rename old index from '' to Date
    data.columns = ['Date', *data.columns[1:]]
    # Add ticker to dataframe
    data['Ticker'] = ticker.ticker
    dfs.append(data)
    time.sleep(1)


data.iloc[:, :3]  # for display purposes


parser = pd.io.parsers.base_parser.ParserBase({'usecols': None})

for df in dfs:
     df.columns = parser._maybe_dedup_names(df.columns)
df = pd.concat(dfs, ignore_index=True)
df = df.set_index(['Ticker','Date'])
df.iloc[:,:5] # for display purposes
df = df.reset_index()
df
####################

print(df.shape)
print(df.tail(10))

df.to_excel('../data/sp500_fi.xlsx', engine='openpyxl')
print('End time: ', now)


import pandas as pd
df = pd.read_excel('./data/sp500_fi_.xlsx', engine='openpyxl')
df = df.drop(['Unnamed: 0'], axis=1)
df_ = df.iloc[:,:3]




#
# # 수익성 분석
#
# 회사가 얼마나 벌고 있는지를 평가한다.
#
# 총자산순이익율 (ROA) = (당기순이익/순자산)*100
df_['ROA'] = df['Net Income'] / df['Total Assets']


# 자기자본순이익율 (ROE) = (당기순이익/자기자본)*100
df_['ROE'] = df['Net Income'] / (df['Total Assets'] - df['Total Liab'])

# 매출액영업이익율 = (영업이익 / 매출액)*100
df_['Operating profit ratio'] = df['Operating Income'] / df['Total Revenue']

# 매출액순이익율 = (당기순이익 / 매출액)*100
df_['Profit margin'] = df['Net Income'] / df['Total Revenue']




# 총자산대비영업현금흐름비율 = (영업활동으로 인한 현금흐름 / 총자산 )*100
df_['Cashflow to total assets'] = df['Total Cash From Operating Activities'] / df['Total Assets']



#
# # 안정성 분석
#
# 회사가 영업에 필요한 현금능력은 충분한 부채를 갚지 못해 도산할 위험은 없는지를 나타낸다.
#
# 유동비율 = (유동자산/유동부채)*100
#
df_['Current ratio'] = df['Total Current Assets'] / df['Total Current Liabilities']

# 당좌비율 = (당좌자산/유동부채)*100

df_['Quick ratio'] = (df['Total Current Assets'] - df['Inventory']) / df['Total Current Liabilities']
# 부채비율 = (부채총계 / 자기자본)*100
df_['Debt ratio'] = df['Total Liab'] / df['Total Assets']

# 순차입금의존도 = (순차입금 / 총자산)*100
df_['Net borrowings to total assets ratio'] = df['Net Borrowings'] / df['Total Assets']

# 영업이익대비 이자보상배율 = (영업이익 / 이자비용) (배)
df_['Interest earned ratio'] = df['Operating Income'] / df['Interest Expense']

# # 활동성 분석
#
# 투자한 자산을 적절히 활용하고 있는지를 나타낸다
#
# 총자산회전율 (자본회전율) = (매출액/총자산) (회)
df_['Total assets turnover'] = df['Total Revenue'] / df['Total Assets']

# 재고자산회전율 = (매출액 / 재고자산) (회)
df_['Inventories turnover'] = df['Total Revenue'] / df['Inventory']

# 자산회전율 = 매출액 / (기초총자산+기말총자산)/2 (회)

df_

df_.to_excel('./data/sp500_fi_ratio.xlsx', engine='openpyxl')

