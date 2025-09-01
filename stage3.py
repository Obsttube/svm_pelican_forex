#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_SVM_and_save(df):
    print('Calculating sliding window...')
    X = sliding_window_view(df['close'].values, window_shape=100)
    y = df['trend'].iloc[99:].reset_index(drop=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'stage3_scaler.pkl')
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    
    i=1
    for train_indices, val_indices in kf.split(X_scaled, y):
        print(f'Training on fold {i}/5...')

        X_train, X_val = X_scaled[train_indices], X_scaled[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        svc_model = SVC(kernel='linear', C=1000)
        svc_model.fit(X_train, y_train)

        y_pred = svc_model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, average='macro'))
        recalls.append(recall_score(y_val, y_pred, average='macro'))

        i+=1

    print("Avg accuracy:", np.mean(accuracies))
    print("Avg precision:", np.mean(precisions))
    print("Avg recall:", np.mean(recalls))

    joblib.dump(svc_model, 'stage3_model.pkl')

def load_SVM_and_predict(df):
    X = sliding_window_view(df['close'].values, window_shape=100)

    scaler = joblib.load('stage3_scaler.pkl')
    X_scaled = scaler.transform(X)

    svc_model = joblib.load('stage3_model.pkl')
    new_predictions = svc_model.predict(X_scaled)
    return new_predictions

def classify_markets(df):
    new_preds = load_SVM_and_predict(df)
    
    preds_windows = sliding_window_view(new_preds, window_shape=100)
    
    weights = np.arange(1, 101) # 1...100
    # weighted moving average of last 100 predictions
    wma = np.dot(preds_windows, weights) / weights.sum()
    
    # First classified market is based on first 100 SVM outputs,
    # first of which is based on first 100 raw values.
    # 100th SVM output is based on raw values at positions 99 - 198.
    # First classifed market is at the last raw value position, so 198.
    # Drop all previous rows.
    df.drop(df.index[:198], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.drop('trend', axis=1, inplace=True)
    
    # (-1 - bear, 0 - sideways, 1 - bull)
    df['classified'] = np.where(wma > 0.5, 1, np.where(wma < -0.5, -1, 0))

def load_SVM_and_calculate_dataset(df, output_filename):
    print('Classifying markets...')
    classify_markets(df)
    print(f'Saving dataset...')
    df.to_pickle(output_filename)
    print(f'Saved at {output_filename}.')

def do_stage3():
    df = pd.read_pickle('dataset_stage2.pkl')
    df_test = pd.read_pickle('dataset_test_stage2.pkl')
    train_SVM_and_save(df)
    load_SVM_and_calculate_dataset(df, 'dataset_stage3.pkl')
    load_SVM_and_calculate_dataset(df_test, 'dataset_test_stage3.pkl')

if __name__ == "__main__":
    do_stage3()

