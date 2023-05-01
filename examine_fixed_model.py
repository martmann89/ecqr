import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import data_loaders
import utils
import data_preprocessing
import numpy as np
import pandas as pd

from cp_single import CQR
train_df, val_df, test_df = data_loaders.get_gas_data([0.8,0.2],1,"old",165)
#train_df, val_df, test_df = data_loaders.get_solar_data()
print(len(test_df))

B = 1                      # number of ensembles
alpha = 0.05                # confidence level            
quantiles = [alpha/2,       # quantiles to predict
             0.5,
             1-(alpha/2)] 

# rf hyperparams
n_trees = 20                # number of trees in each rf model

# lstm tcn only
regression = 'quantile'     # options: {'quantile', 'linear'}. If 'linear', just set one quantile
l2_lambda = 1e-4            # weight of l2 regularization in the lstm and tcn models
batch_size = 16             # size of batches using to train the lstm and tcn models

# lstm hyperparams
units = 32           # number of units in each lstm layer
n_layers = 1              # number of lstm layers in the model

# units = 64
# n_layers = 1


# tcn hyperparams
dilations = [1,2]#,4,8]       # dilation rate of the Conv1D layers
n_filters = 256             # filters in each Conv1D layer 
kernel_size = 7             # kernel size in each ConvID layer

learning_rate = 1e-3
time_steps_in = 6


P = {'B':B, 'alpha':alpha, 'quantiles':quantiles,
                            'n_trees':n_trees,  
                            'regression':regression,'l2':l2_lambda, 'batch_size':batch_size,
                            'units':units,'n_layers':n_layers,
                            'dilations':dilations, 'n_filters':n_filters, 'kernel_size':kernel_size,
                            'learning_rate' : learning_rate, 'time_steps_in' : time_steps_in}
P['model_type'] = 'tcn'

if P['model_type'] == 'tcn':
    colnames_stats = ['alpha', 'batch_size', 'kernel_size', 'dilations','n_filters', 'l2_lambda', 'learning_rate','epochs', 'time_steps_in', 'coverage', 'avg_width','avg_width_transformed', 'cwc','coverage_conf', 'avg_width_conf', 'avg_width_transformed_conf', 'cwc_conf']# ['Nof Learners', 'alpha', 'batch_size', 'units', 'n_layers', 'l2_lambda', 'learning_rate','epochs', 'time_steps_in', 'coverage', 'avg_width','avg_width_transformed', 'cwc','coverage_conf', 'avg_width_conf', 'avg_width_transformed_conf', 'cwc_conf']

colnames_result = ['low','high','mean']
base_path = './predictions/165d_TTF_FM_old/tcn2/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

nof_runs = 10
train_data, val_x, val_y, test_x, test_y, Scaler = data_preprocessing.data_windowing(df=train_df, 
                                                                                    val_data=val_df,
                                                                                    test_data=test_df,
                                                                                    B=1, 
                                                                                    time_steps_in=time_steps_in, 
                                                                                    time_steps_out=1, 
                                                                                    label_columns=['Price'])

P['n_vars'] = test_x.shape[2] 
P['time_steps_out'] = test_y.shape[1]
csv_path_statistics = base_path + 'stats.csv'
for i in range(1,nof_runs+1):
    coverage_sum = 0
    avg_length_sum = 0
    avg_length_trans_sum = 0
    cwc_sum = 0
    coverage_conf_sum = 0
    avg_length_conf_sum = 0 
    avg_length_trans_conf_sum = 0 
    cwc_conf_sum = 0

    PI, PI_conf, hist = CQR(train_data,val_x,val_y,test_x, test_y, P, plot_training=False)

    coverage, avg_length, avg_length_trans, cwc = utils.compute_coverage_len(test_y.flatten(), PI[:,:,0].flatten(), PI[:,:,2].flatten(), Scaler, verbose=False)
    print(f"PI coverage: {coverage*100:.4f}%, PI avg. length: {avg_length:.4f}, PI avg. length transformed: {avg_length_trans:.4f}, CWC: {cwc:.4f}")

    coverage_conf, avg_length_conf, avg_length_trans_conf, cwc_conf = utils.compute_coverage_len(test_y.flatten(), PI_conf[:,:,0].flatten(), PI_conf[:,:,2].flatten(), Scaler, verbose=False)
    print(f"PI conformalized coverage: {coverage_conf*100:.4f}%, PI avg. length: {avg_length_conf:.4f}, PI avg. length transformed: {avg_length_trans_conf:.4f}, CWC: {cwc_conf:.4f}")

    coverage_sum += coverage
    avg_length_sum += avg_length
    avg_length_trans_sum += avg_length_trans 
    cwc_sum += cwc

    coverage_conf_sum += coverage_conf
    avg_length_conf_sum += avg_length_conf
    avg_length_trans_conf_sum += avg_length_trans_conf 
    cwc_conf_sum += cwc_conf
    
    csv_paths = [base_path + f'intervals_prevs_{time_steps_in}_run_{i}_conf.csv',
                 base_path + f'intervals_prevs_{time_steps_in}_run_{i}.csv']
    result_dfs = [pd.DataFrame({'low' : PI_conf[:,:,0].flatten(), 'high' : PI_conf[:,:,2].flatten(), 'mean' : PI_conf[:,:,1].flatten()}),
                  pd.DataFrame({'low' : PI[:,:,0].flatten(), 'high' : PI[:,:,2].flatten(), 'mean' : PI[:,:,1].flatten()})]
    for idx,csv_path in enumerate(csv_paths):
        result_dfs[idx].to_csv(csv_path, sep=";")

    utils.plot_PIs(test_y, PI[:,:,1],
                    PI[:,:,0], PI[:,:,2],
                    PI_conf[:,:,0], PI_conf[:,:,2],
                    scaler=Scaler, title=f'Single CQR - {P["model_type"]}',save_path=base_path + f'run_{i}')
    
if P['model_type'] == 'tcn':
    stats_df = pd.DataFrame({'alpha' : alpha, 'batch_size' : batch_size, 'kernel_size' : kernel_size, 'dilations': dilations,"n_filters" : n_filters, 'l2_lambda' : l2_lambda,
                                                'learning_rate' : learning_rate, 'time_steps_in' : time_steps_in,'coverage' : coverage_sum / nof_runs, 'avg_width' : avg_length_sum / nof_runs, 'avg_width_transformed' : avg_length_trans_sum / nof_runs, 
                                                'cwc' : cwc_sum / nof_runs, 'coverage_conf' : coverage_conf_sum / nof_runs, 'avg_width_conf' : avg_length_conf_sum / nof_runs, 'avg_width_transformed_conf' : avg_length_trans_conf_sum / nof_runs, 'cwc_conf' : cwc_conf_sum / nof_runs})
stats_df.to_csv(csv_path_statistics, sep=";")