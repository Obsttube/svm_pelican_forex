#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import pickle
import random as r
import sys
import time


# returns: 0 - do nothing, 1 - buy, 2 - sell
def decide_action(close_price, rsi, rsi_w, roc, roc_w, sma, sma_w, ema, ema_w,
                  wma, wma_w, macd, macd_w, K):
    voter_buy = 0
    voter_sell = 0
    
    if rsi < 30:
        voter_buy += rsi_w
    elif rsi > 70:
        voter_sell += rsi_w
    
    if roc > 0:
        voter_buy += roc_w
    elif roc < 0:
        voter_sell += roc_w
    
    if sma < close_price:
        voter_buy += sma_w
    elif sma > close_price:
        voter_sell += sma_w
    
    if ema < close_price:
        voter_buy += ema_w
    elif ema > close_price:
        voter_sell += ema_w
    
    if wma < close_price:
        voter_buy += wma_w
    elif wma > close_price:
        voter_sell += wma_w
    
    if macd > 0:
        voter_buy += macd_w
    elif macd < 0:
        voter_sell += macd_w
    
    if voter_buy > K * voter_sell:
        return 1 # buy
    elif voter_sell > K * voter_buy:
        return 2 # sell
    else:
        return 0 # do nothing

def gen_pelican(params_max):
    params = []
    for _ in range(3):
        for i in range(5):
            params.append(r.randint(0, params_max[i]))
            params.append(r.uniform(0, 1))
        params.append(r.randint(0, params_max[5]))
        params.append(r.randint(0, params_max[6]))
        params.append(r.uniform(0, 1))
    return params

def calculate_roi(df, params, K, L):
    # params = [rsi0, rsi0_w, roc0, roc0_w, sma0, sma0_w, ema0, ema0_w, 
    # wma0, wma0_w, macd0_fast, macd0_slow, macd0_w,
    # rsi1, rsi1_w, roc1, roc1_w, sma1, sma1_w, ema1, ema1_w,
    # wma1, wma1_w, macd1_fast, macd1_slow, macd1_w,
    # rsi2, rsi2_w, roc2, roc2_w, sma2, sma2_w, ema2, ema2_w,
    # wma2, wma2_w, macd2_fast, macd2_slow, macd2_w]
    
    # initial balance 100 000 EUR + 0 USD
    initial_balance_eur = 100000
    
    balance_eur = initial_balance_eur
    balance_usd = 0
    position = 0 # 0 - none, 1 - long, 2 - short
    leverage = 1
    margin = 0
    
    for row in df.itertuples(index=False):
        market = row.classified # bear (-1 - bear, 0 - sideways, 1 - bull)
        start_idx = 0 # bear
        if market == 0:
            start_idx = 13 # sideways
        elif market == 1:
            start_idx = 26 # bull
        # macd_fast + macd_slow
        macd_idx = params[start_idx+10] * 16 + params[start_idx+11]
        action = decide_action(row.close,
                               row.rsi[params[start_idx+0]], params[start_idx+1],
                               row.roc[params[start_idx+2]], params[start_idx+3],
                               row.sma[params[start_idx+4]], params[start_idx+5],
                               row.ema[params[start_idx+6]], params[start_idx+7],
                               row.wma[params[start_idx+8]], params[start_idx+9],
                               row.macd[macd_idx], params[start_idx+12], K)
        # action 0 - do nothing, 1 - buy, 2 - sell
        if position == 0 and action == 1:
            if balance_usd > 0:
                position = 1
                if (market == 1 and row.SMA_low < row.low
                    and row.SMA_high < row.high):
                    leverage = L
                    margin = balance_usd
                else:
                    leverage = 1
                balance_eur = balance_usd  * leverage / row.close
                balance_usd = 0
        elif position == 0 and action == 2:
            if balance_eur > 0:
                position = 2
                if (market == -1 and row.SMA_low > row.low
                    and row.SMA_high > row.high):
                    leverage = L
                    margin = balance_eur
                else:
                    leverage = 1
                balance_usd = balance_eur * leverage * row.close
                balance_eur = 0
        elif position == 1 and action == 2:
            if balance_eur > 0:
                position = 2
                balance_usd = balance_eur * row.close - ((leverage - 1) * margin)
                balance_eur = 0
                margin = balance_usd / row.close
                if (market == -1 and row.SMA_low > row.low
                    and row.SMA_high > row.high):
                    leverage = L
                    balance_usd *= leverage
                else:
                    leverage = 1
        elif position == 1 and action == 0:
            if balance_eur > 0:
                position = 0
                balance_usd = balance_eur * row.close - ((leverage - 1) * margin)
                balance_eur = 0
                leverage = 1
        elif position == 2 and action == 1:
            if balance_usd > 0:
                position = 1
                balance_eur = balance_usd / row.close - ((leverage - 1) * margin)
                balance_usd = 0
                margin = balance_eur * row.close
                if (market == 1 and row.SMA_low < row.low
                    and row.SMA_high < row.high):
                    leverage = L
                    balance_eur *= leverage
                else:
                    leverage = 1
        elif position == 2 and action == 0:
            if balance_usd > 0:
                position = 0
                balance_eur = balance_usd / row.close - ((leverage - 1) * margin)
                balance_usd = 0
                leverage = 1
    
    if balance_eur == 0 and balance_usd != 0:
        balance_eur = (balance_usd / df.loc[df.index[-1], 'close']
                       - (leverage - 1) * margin)
    
    roi = (balance_eur - initial_balance_eur) / initial_balance_eur
    return roi

def gen_population(params_max, size):
    population = []
    for _ in range(size):
        population.append(gen_pelican(params_max))
    return population

def equation7a(pelican, prey, i, I, max_val, rounding=True):
    b = prey[i] - I * pelican[i]
    if b >= 0:
        pelican[i] += r.uniform(0, min(max_val - pelican[i], b))
    else:
        pelican[i] += r.uniform(max(-pelican[i], b), 0)
    if rounding:
        pelican[i] = round(pelican[i])
    if pelican[i] < 0:
        pelican[i] = 0
    elif pelican[i] > max_val:
        pelican[i] = max_val

def equation7b(pelican, prey, i, I, max_val, rounding=True):
    b = pelican[i] - prey[i]
    if b >= 0:
        pelican[i] += r.uniform(0, min(max_val - pelican[i], b))
    else:
        pelican[i] += r.uniform(max(-pelican[i], b), 0)
    if rounding:
        pelican[i] = round(pelican[i])
    if pelican[i] < 0:
        pelican[i] = 0
    elif pelican[i] > max_val:
        pelican[i] = max_val

def equation7(eq7_variant, pelican, prey, I, params_max):
    for j in range(3):
            for i in range(5):
                eq7_variant(pelican, prey, j*13 + i*2, I, params_max[i])
                eq7_variant(pelican, prey, j*13 + i*2+1, I, 1, False)
            eq7_variant(pelican, prey, j*13 + 10, I, params_max[5])
            eq7_variant(pelican, prey, j*13 + 11, I, params_max[6])
            eq7_variant(pelican, prey, j*13 + 12, I, 1, False)

def phase1(df, population, prey, K, L, params_max):
    prey_roi = calculate_roi(df, prey, K, L)
    for i in range(len(population)):
        new_pelican = population[i].copy()
        pelican_roi = calculate_roi(df, new_pelican, K, L)
        I = r.randint(1, 2)
        # The following if statement is reversed when compared to the one in
        # Trojovský et al., because a goal function is used here instead of
        # a cost function. Roi is maximised here, not minimised.
        if pelican_roi < prey_roi:
            equation7(equation7a, new_pelican, prey, I, params_max)
        else:
            equation7(equation7b, new_pelican, prey, I, params_max)
        new_pelican_roi = calculate_roi(df, new_pelican, K, L)
        # The following if statement is reversed when compared to the one in
        # Trojovský et al., because a goal function is used here instead of
        # a cost function. Roi is maximised here, not minimised.
        if new_pelican_roi > pelican_roi:
            population[i] = new_pelican

def equation8a(pelican, prey, i, max_val, t, T, rounding=True):
    tmp = 0.2 * (1-t/T) * pelican[i]
    pelican[i] += r.uniform(0, min(max_val - pelican[i], tmp))
    if rounding:
        pelican[i] = round(pelican[i])
    if pelican[i] < 0:
        pelican[i] = 0
    elif pelican[i] > max_val:
        pelican[i] = max_val

def equation8(pelican, prey, t, T, params_max):
    for j in range(3):
            for i in range(5):
                equation8a(pelican, prey, j*13 + i*2, params_max[i], t, T)
                equation8a(pelican, prey, j*13 + i*2+1, 1, t, T, False)
            equation8a(pelican, prey, j*13 + 10, params_max[5], t, T)
            equation8a(pelican, prey, j*13 + 11, params_max[6], t, T)
            equation8a(pelican, prey, j*13 + 12, 1, t, T, False)

def phase2(df, population, prey, t, T, K, L, params_max):
    best_roi = float('-inf')
    worst_roi = float('inf')
    roi_sum = 0
    for i in range(len(population)):
        new_pelican = population[i].copy()
        pelican_roi = calculate_roi(df, new_pelican, K, L)
        equation8(new_pelican, prey, t, T, params_max)
        new_pelican_roi = calculate_roi(df, new_pelican, K, L)
        # The following if statement is reversed when compared to the one in
        # Trojovský et al., because a goal function is used here instead of
        # a cost function. Roi is maximised here, not minimised.
        if new_pelican_roi > pelican_roi:
            population[i] = new_pelican
            if new_pelican_roi > best_roi:
                best_roi = new_pelican_roi
            if new_pelican_roi < worst_roi:
                worst_roi = new_pelican_roi
            roi_sum += new_pelican_roi
        else:
            if pelican_roi > best_roi:
                best_roi = pelican_roi
            if pelican_roi < worst_roi:
                worst_roi = pelican_roi
            roi_sum += pelican_roi
    avg_roi = roi_sum/len(population)
    return best_roi, avg_roi, worst_roi

def get_population_rois(df, population, K, L):
    rois = []
    for pelican in population:
        rois.append(calculate_roi(df, pelican, K, L))
    return np.array(rois)

def do_stage4a(df_train, df_val):
    print('Started.')
    start_time = time.time()
    params_max = [25, 15, 90, 90, 90, 10, 15]

    population_size = 20
    num_iterations = 500
    K = 10000
    L = 10

    population = gen_population(params_max, population_size)

    for i in range(num_iterations):
        prey = gen_pelican(params_max)
        phase1(df_train, population, prey, K, L, params_max)
        best_roi, avg_roi, worst_roi = phase2(df_train, population, prey, i,
                                              num_iterations, K, L, params_max)
        print(f'{i}/{num_iterations} best: {best_roi} avg: {avg_roi} worst: {worst_roi}')

    print('\nROI of each pelican on training dataset:')
    rois_train = get_population_rois(df_train, population, K, L)
    for roi in rois_train:
        print(roi)

    print('\nBest ROI:')
    print(rois_train.max())
    print('Average ROI:')
    print(rois_train.mean())
    print('Worst ROI:')
    print(rois_train.min())

    print('\nROI of each pelican on validation dataset:')
    rois_val = get_population_rois(df_val, population, K, L)
    for roi in rois_val:
        print(roi)
    
    print('\nBest ROI:')
    print(rois_val.max())
    print('Average ROI:')
    print(rois_val.mean())
    print('Worst ROI:')
    print(rois_val.min())
    
    print('\nAverage ROI of top 20% pelicans on training dataset:')
    sorted_indices = np.argsort(rois_train)[::-1]
    top_20_percent_indices = sorted_indices[:round(population_size*0.2)]
    print(rois_train[top_20_percent_indices].mean())
    print('\nAverage ROI of top 20% pelicans from training dataset, '
          + 'on validation dataset:')
    print(rois_val[top_20_percent_indices].mean())
    
    print(f'\nTime: {(time.time()-start_time)/60/60} h')

    with open('population.pkl', 'wb') as f:
        pickle.dump(population, f)

def do_stage4():
    df = pd.read_pickle('dataset_stage3.pkl')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_train = df[df['datetime'] < pd.Timestamp('2013-01-01 00:00:00')]
    df_val = df[df['datetime'] >= pd.Timestamp('2013-01-01 00:00:00')]
    do_stage4a(df_train, df_val)

def do_stage4_test():
    df_train = pd.read_pickle('dataset_stage3.pkl')
    df_test = pd.read_pickle('dataset_test_stage3.pkl')
    do_stage4a(df_train, df_test)

if __name__ == "__main__":
    do_stage4()

