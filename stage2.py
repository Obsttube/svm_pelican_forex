#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import ta
from scipy.stats import linregress


def calculate_trend(prices):
    x = range(len(prices))
    slope_threshold = 0.01 * prices.mean() / 100
    slope, _, _, _, _ = linregress(x, prices)
    if slope > slope_threshold:
        return 1 # bull
    elif slope < -slope_threshold:
        return -1 # bear
    else:
        return 0 # sideways

def add_technical_indicators(filename, output_filename):
    print(f'Calculating technical indicators for {filename}...')
    df = pd.read_csv(filename)
    print('Calculating rsi...')
    df['rsi'] = list(np.column_stack([
        ta.momentum.RSIIndicator(close=df['close'], window=w).rsi().values
        for w in range(5, 31)
    ]))
    print('Calculating roc...')
    df['roc'] = list(np.column_stack([
        ta.momentum.ROCIndicator(close=df['close'], window=w).roc().values
        for w in range(15, 31)
    ]))
    print('Calculating sma...')
    df['sma'] = list(np.column_stack([
        ta.trend.SMAIndicator(close=df['close'], window=w).sma_indicator().values
        for w in range(10, 101)
    ]))
    print('Calculating ema...')
    df['ema'] = list(np.column_stack([
        ta.trend.EMAIndicator(close=df['close'], window=w).ema_indicator().values
        for w in range(10, 101)
    ]))
    print('Calculating wma... (approx. 35 mins)')
    wma_arrays = []
    for w in range(10, 101):
        print(f'{w}/100')
        wma = ta.trend.WMAIndicator(close=df['close'], window=w).wma().values
        wma_arrays.append(wma)
    df['wma'] = list(np.column_stack(wma_arrays))
    print('Calculating macd...')
    df['macd'] = list(np.column_stack([
        ta.trend.MACD(close=df['close'],
                      window_fast=f, window_slow=s).macd().values
        for f in range(10, 21)
        for s in range(20, 36)
        # Although in practice f: 0-10 indices for range(10, 21),
        # s: 0-15 indices for range(20, 36), here is an example for f: 0-2; s: 0-2:
        # f[0] s[0]; f[0] s[1]; f[0] s[2]; f[1] s[0]; f[1] s[1]; f[1] s[2]; f[2] s[0]; f[2] s[1];
        # f[2] s[2]
    ]))
    print('Calculating trend...')
    df['trend'] = df['close'].rolling(window=100).apply(calculate_trend, raw=False)
    print('Calculating SMA of period H=3...')
    H=3
    df[f'SMA_low'] = df['low'].rolling(window=H).mean()
    df[f'SMA_high'] = df['high'].rolling(window=H).mean()
    print(f'Saving dataset...')
    df.to_pickle(output_filename)
    print(f'Saved at {output_filename}.')

def do_stage2():
    add_technical_indicators('dataset_stage1.txt', 'dataset_stage2.pkl')
    add_technical_indicators('dataset_test_stage1.txt', 'dataset_test_stage2.pkl')

if __name__ == "__main__":
    do_stage2()

