import time

from sec_api import ExtractorApi
from sec_api import QueryApi
import pandas as pd

API_KEY = 'API_KEY_012345'
extractorApi = ExtractorApi(API_KEY)

list_sp = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
ticker_list = list_sp['Symbol'].tolist()

queryApi = QueryApi(api_key=API_KEY)

text_list = []


i = 0

for i in range(0, len(list_sp), 1):
    time.sleep(1)
    if i % 10 == 0:
        print("Processing...: ", i)

    ticker = list_sp.iloc[i, 0]
    query = {
        "query": {"query_string": {
            "query": "ticker:" + ticker + " AND filedAt:{2017-01-01 TO 2017-12-31} AND formType:\"10-K\""
        }},
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}]
    }


    filings = queryApi.get_filings(query)
    dic_fil = filings['filings']
    if len(dic_fil) > 0:
        dic = dic_fil[0]

        text1 = extractorApi.get_section(dic['linkToFilingDetails'], "1", "text")
        text2 = extractorApi.get_section(dic['linkToFilingDetails'], "1A", "text")
        text3 = extractorApi.get_section(dic['linkToFilingDetails'], "1B", "text")
        text4 = extractorApi.get_section(dic['linkToFilingDetails'], "2", "text")
        text5 = extractorApi.get_section(dic['linkToFilingDetails'], "3", "text")
        text6 = extractorApi.get_section(dic['linkToFilingDetails'], "4", "text")
        text7 = extractorApi.get_section(dic['linkToFilingDetails'], "5", "text")
        text8 = extractorApi.get_section(dic['linkToFilingDetails'], "6", "text")
        text9 = extractorApi.get_section(dic['linkToFilingDetails'], "7", "text")
        text10 = extractorApi.get_section(dic['linkToFilingDetails'], "7A", "text")
        text11 = extractorApi.get_section(dic['linkToFilingDetails'], "8", "text")
        text12 = extractorApi.get_section(dic['linkToFilingDetails'], "9", "text")
        text13 = extractorApi.get_section(dic['linkToFilingDetails'], "9A", "text")
        text14 = extractorApi.get_section(dic['linkToFilingDetails'], "9B", "text")
        text15 = extractorApi.get_section(dic['linkToFilingDetails'], "10", "text")
        text16 = extractorApi.get_section(dic['linkToFilingDetails'], "11", "text")
        text17 = extractorApi.get_section(dic['linkToFilingDetails'], "12", "text")
        text18 = extractorApi.get_section(dic['linkToFilingDetails'], "13", "text")
        text19 = extractorApi.get_section(dic['linkToFilingDetails'], "14", "text")

        text_list.append([list_sp.iloc[i, 0], list_sp.iloc[i, 1], text1, text2, text3, text4, text5, text6, text7, text8, text9, text10,
                          text11, text12, text13, text14, text15, text16, text17, text18, text19])

sec_code = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14"]
# len(sec_code)


df_text_list = pd.DataFrame(text_list, columns=["ticker", "companynames", "1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14"])
df_text_list.to_excel('../data/2017_10k.xlsx', engine='openpyxl')
