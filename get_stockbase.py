'''
模块说明：快速获取A股市场的即时与历史行情数据，获得即时的指数成分股信息
接口：
1、实例生成
AmarketData(index_code:可不写=None,start=None,end=None,dir=None)         实例传入起止时间，是否传入文件目录保存看实际需求
2、全市场即时数据【查看/提取】
AmarketData.get_spot(csv_name =None)                                    若方法传入csv文件名，实例已传入文件保存目录，行情开始写入csv文件
3、特定代码历史数据 【查看/提取】
AmarketData.get_hist(index_code,period="daily",csv_name =None)          传入csv文件名，实例已传入文件保存
3、指数成分股数据【查看/提取】
AmarketData.get_index_spot(symbol,requirements)                         若方法传入csv文件名，实例传入文件保存目录，行情写入csv文件
AmarketData.get_index_hist(self,symbol)                                 若方法传入指数代码，实例已传入文件保存目录，行情写入csv文件
'''

import sys
sys.path.append("D:/programfiles/anaconda/envs/pytorch/lib/site-packages")
import os
import akshare as ak
import pandas as pd
from datetime import datetime
import requests
import tushare as ts
import time
xxxx=[i for i in ak.__dict__ if i.startswith('con')]
print(pd.DataFrame(xxxx))
# import importlib
# importlib.reload(ak)

def index_zh_a_hist(
    symbol: str = "000859",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "22220101",
) -> pd.DataFrame:
    """
    东方财富网-中国股票指数-行情数据
    https://quote.eastmoney.com/zz/2.000859.html
    :param symbol: 指数代码
    :type symbol: str
    :param period: choice of {'daily', 'weekly', 'monthly'}
    :type period: str
    :param start_date: 开始日期
    :type start_date: str
    :param end_date: 结束日期
    :type end_date: str
    :return: 行情数据
    :rtype: pandas.DataFrame
    """
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": f"1.{symbol}",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": period_dict[period],
        "fqt": "0",
        "beg": "0",
        "end": "20500000",
        "_": "1623766962675"
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    if data_json["data"] is None:
        params = {
            "secid": f"0.{symbol}",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": period_dict[period],
            "fqt": "0",
            "beg": "0",
            "end": "20500000",
            "_": "1623766962675"
        }
        r = requests.get(url, params=params)
        data_json = r.json()
        if data_json["data"] is None:
            params = {
                "secid": f"2.{symbol}",
                "ut": "7eea3edcaed734bea9cbfc24409ed989",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": period_dict[period],
                "fqt": "0",
                "beg": "0",
                "end": "20500000",
                "_": "1623766962675"
            }
            r = requests.get(url, params=params)
            data_json = r.json()
            if data_json["data"] is None:
                params = {
                    "secid": f"47.{symbol}",
                    "ut": "7eea3edcaed734bea9cbfc24409ed989",
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                    "klt": period_dict[period],
                    "fqt": "0",
                    "beg": "0",
                    "end": "20500000",
                    "_": "1623766962675"
                }
    r = requests.get(url, params=params)
    data_json = r.json()
    temp_df = pd.DataFrame(
        [item.split(",") for item in data_json["data"]["klines"]]
    )
    temp_df.columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    temp_df.index = pd.to_datetime(temp_df["日期"], errors="coerce")
    temp_df = temp_df[start_date:end_date]
    temp_df.reset_index(inplace=True, drop=True)
    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"], errors="coerce")
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    return temp_df

#以下代码使用迭代取出*requirements的所有变量放入列表
def loop_csv(input):
    cols1 = []
    for i in input:
        if isinstance(i,str):
            cols1.append(i)
        elif isinstance(i,tuple) or isinstance(i,list):
            AmarketData.loop_csv(i)
        else:
            raise TypeError
    return cols1



class AmarketData:
    def __init__(index_code=None,start=None,end=None,dir=None):
        AmarketData.date_time = datetime.now().strftime('%Y%m%d')
        AmarketData.index_code = index_code
        AmarketData.start_time = start
        AmarketData.end_time = end
        AmarketData.dir = dir
        print('\n\n'+f'Amarketdata实例生成成功！\n读取数据的期限范围为{AmarketData.start_time}-{AmarketData.end_time}\n需要保存的数据会保存在目录{AmarketData.dir}下\n今日时间是{AmarketData.date_time}\n')
           
    '''全A股市场中的股票即时数据'''
    def get_spot(self,csv_name =None):
        share_list =  ak.stock_zh_a_spot_em()
        if csv_name:   
            if self.dir:
                print('get_spot方法中已传入csv文件名，实例已传入文件保存目录,行情开始写入','*'*10)
                wholename = os.path.join(AmarketData.dir,csv_name)
                x = open(f'{wholename}', 'w')
                share_list.to_csv(x, index=False)   
                x.close()
                print(f"文件{csv_name}已成功保存至目录{self.dir}下，保存日期为{AmarketData.date_time}：\n")
            else:
                raise Exception("请设置实例生成文件的保存路径（get_spot方法中已传入csv文件名）")  
        else:
            print('get_spot方法中未传入csv文件名，显示当前A股行情，并已保存为df格式')
            print(f"{share_list}")  
        return share_list

    '''指定指数/股票的历史行情数据'''
    def get_hist(self,index_code,period="daily",csv_name =None):
        # 获取沪深300指数的历史数据
        index_data=ak.index_zh_a_hist(symbol=index_code,period=period,start_date=AmarketData.start_time,end_date=AmarketData.end_time)
        if csv_name:
            if self.dir:
                    print('get_hist方法中已传入csv文件名，实例已传入文件保存目录,历史行情开始写入','*'*10)
                    wholename = os.path.join(AmarketData.dir,csv_name)
                    x = open(f'{wholename}', 'w')
                    index_data.to_csv(x, index=False)   
                    x.close()
                    print(f"{index_code}在{AmarketData.start_time}到{AmarketData.start_time}间的历史行情数据已成功保存至目录{self.dir}下：\n")
            else:
                raise Exception("请设置实例生成文件的保存路径（get_spot方法中已传入csv文件名）")  
        else:
            print('get_hist方法中未传入csv文件名，显示当前A股行情，并已保存为df格式')
            print(f'以下是代码为{index_code}在{AmarketData.start_time}到{AmarketData.start_time}间的历史行情数据（完整文件已保存为df数组）:\n\n')
        return index_data
    def get_index_spot(self,symbol,col=None):
        index_df = ak.index_stock_cons_sina(symbol=symbol)
        if not symbol:
            print(f'以下是指数代码为{symbol}的成分股信息：\n')
            print(index_df)
        else:
            if index_df.empty:
                print('ip被封了，请更换IP代理')
            else:
                print('\n'+f'get_index方法已接收到symbol参数:{symbol}，服务器正常访问，正在保存................................')
                xname =str(os.path.join(AmarketData.dir,symbol))+'const'+'_'+AmarketData.date_time+'.csv'
                x=open(xname, 'w')
                index_df.to_csv(x, index=False)
                x.close()
                print(os.path.join(AmarketData.dir,symbol))
                print(f'{symbol}指数于{AmarketData.date_time}的成分股数据已成功保存至目录{AmarketData.dir}下\n\n')
                if col:
                    print(f'get_index方法已接收特定列名{col}，正在处理.................')
                    cols_headers = loop_csv(col)
                    pd_i1 = pd.read_csv(xname)
                    pd_i1 = pd_i1[cols_headers]
                    col_name = str(os.path.join(AmarketData.dir,symbol))+symbol+'_'+AmarketData.date_time+'_col_'+'_'.join(cols_headers)+'.csv'
                    with open(col_name,'w') as fout:
                        pd_i1.to_csv(fout,index=False)
                        print(f'{symbol}指数于{AmarketData.date_time}的成分股数据已成功保存至目录{AmarketData.dir}下\n\n')
                else:
                    print('get_index方法未接收col列名参数，任务结束')
    def get_index_hist(self,symbol):
        index_df = ak.index_stock_cons_sina(symbol='000300')
        df = index_df[['code','name']]
        df1 =df.iloc[:,[0,1]].values
        #print(df1)
        time.sleep(2)
        print('\n\n'+f'开始下载代码为{symbol}的指数所有成分股于{AmarketData.start_time}-{AmarketData.end_time}期间的历史数据')
        readdir = os.listdir(AmarketData.dir)
        csv_name_lists = [allist for allist in readdir if '.csv' in allist]
        
        
        for element,name in df1:
            x1 = os.path.join(AmarketData.dir,element)
            wholename = x1 + f'_{AmarketData.start_time}_{AmarketData.end_time}'+'.csv'
            checkname = str(element)+f'_{AmarketData.start_time}_{AmarketData.end_time}'+'.csv'
            print(f'正在检查{element}数据是否已存在...................................')
            if checkname in csv_name_lists:
                print(f'{checkname}已存在，开始进行下一个股票的下载')
                continue
            else:
                print(f'{checkname}不存在，准备下载并保存')
                #marketData.get_index_spot(symbol=element)
                print(f'开始下载股票代码:{element},股票名称{name}于{AmarketData.start_time}-{AmarketData.end_time}期间的历史数据...........')
                index_data=index_zh_a_hist(symbol=element,period='daily',start_date=AmarketData.start_time,end_date=AmarketData.end_time)
                print(f'股票代码:{element},股票名称{name}于{AmarketData.start_time}-{AmarketData.end_time}期间的历史数据已下载完成')
                x = open(f'{wholename}', 'w')
                index_data.to_csv(x, index=False)   
                x.close()
                print(f"{element}在{AmarketData.start_time}到{AmarketData.start_time}间的历史行情数据已成功保存至目录{self.dir}下\n")
                print('正在等待中(35s)，防止反爬.......................................\n')
                time.sleep(35)
        print('\n\n'+f'代码为{symbol}的指数所有成分股于{AmarketData.start_time}-{AmarketData.end_time}期间的历史数据已下载完成')
                    
if __name__ == "__main__":
    md =AmarketData(start='20220101',end='20241031',dir=r'D:/something/task/task2_lasso_regression/data_000300')
    #获得当前A股全市场行情数据
    #md.get_spot(csv_name="all_share.csv")
    #获得特定代码股票代码的历史行情数据
    #md.get_hist(index_code="000300",csv_name="sh000300.csv")
    #获得特定指数成分股的信息
    #x = md.get_index(symbol='000300')
    #获得特定指数成分股的任意列信息
    req = ['code','name']
    md.get_index_spot(symbol='000300',col=req)
    #获得特定指数的所有成分股的历史行情数据
    #md.get_index_hist(symbol='000300')
