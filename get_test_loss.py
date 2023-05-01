import numpy as np
import tensorflow as tf
import pandas as pd
import os

def _pin_loss(labels, pred, alpha):
    loss = []
    for i in range(len(labels)):
        error = labels[i] - pred[i]
        if error <= 0:
            loss.append(-error*(1-alpha))
        else:
            loss.append(error*alpha)
    
    return np.mean(loss)

alphas = [0.05,0.95]
label_df = pd.read_csv(f'./TTF_FM_old.csv', sep=";")
label_df = pd.DataFrame(label_df, columns=['Price'])
labels_all = label_df.to_numpy()
path = './predictions/165d_TTF_FM_old/lstm/alpha_0_05/'
csv_files = [f for f in os.listdir(f'{path}') if os.path.isfile(os.path.join(f'{path}', f))]


for alpha in alphas:
    loss_total = []
    for file in csv_files:
        df = pd.read_csv(f'{path}{file}', sep=";")
        results = pd.DataFrame(df, columns=['lower','upper','mean']).to_numpy()
        res_lo = results[:,0]
        res_hi = results[:,1]
        labels = labels_all[(len(labels_all) - results.shape[0]):]
        loss = _pin_loss(labels, results[:,2], alpha)
        loss_total.append(loss)

    loss_total = np.mean(loss_total)
    print(f"alpha:{alpha} ", loss_total)

