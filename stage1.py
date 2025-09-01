#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import os
import pandas as pd
import zipfile


def do_stage1(ticker):
    if not os.path.exists(ticker):
        os.makedirs(ticker)

    filenames = os.listdir(ticker)
    if len(filenames) == 0:
        print('Download yearly ASCII M1 data from HistData.com, '
              + f'place it in {ticker} directory and try again.')
        return

    df_list = []
    for filename in filenames:
        print(f'Processing {filename}...')
        archive = zipfile.ZipFile(ticker + '/' + filename, 'r')
        for filename_in_zip in archive.namelist():
            if os.path.splitext(filename_in_zip)[1] == '.csv':
                print(f'Reading {filename_in_zip}...')
                csv = archive.read(filename_in_zip)
                df_list.append(pd.read_csv(io.BytesIO(csv), sep=';', header=None))
        archive.close()

    print('Concatenating (1-2 mins)...')
    df = pd.concat(df_list, ignore_index=True)
    df[0] = pd.to_datetime(df[0], format="%Y%m%d %H%M%S")
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    print('Grouping...')
    df = df.groupby(df['datetime'].dt.floor('H')).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()

    df1 = df[
        (df['datetime'] >= pd.Timestamp('2003-01-01 00:00:00'))
        & (df['datetime'] < pd.Timestamp('2015-01-01 00:00:00'))
    ]
    df2 = df[
        (df['datetime'] >= pd.Timestamp('2015-01-01 00:00:00'))
        & (df['datetime'] < pd.Timestamp('2016-03-01 00:00:00'))
    ]

    df1.to_csv('dataset_stage1.txt', encoding='utf-8',index=False, header=True)
    df2.to_csv('dataset_test_stage1.txt', encoding='utf-8', index=False, header=True)

    print('Done!')

if __name__ == "__main__":
    do_stage1('EURUSD')

