import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import data_loaders
import utils
import data_preprocessing
import numpy as np
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

train_data, val_x, val_y, test_x, test_y, Scaler = data_preprocessing.data_windowing(df=train_df, 
                                                                                    val_data=val_df,
                                                                                    test_data=test_df,
                                                                                    B=1, 
                                                                                    time_steps_in=time_steps_in, 
                                                                                    time_steps_out=1, 
                                                                                    label_columns=['Price'])
print(len(test_y))
P['n_vars'] = test_x.shape[2] 
P['time_steps_out'] = test_y.shape[1]

PI, PI_conf, hist = CQR(train_data,val_x,val_y,test_x, test_y, P, plot_training=True)

coverage, avg_length, avg_length_trans, cwc = utils.compute_coverage_len(test_y.flatten(), PI[:,:,0].flatten(), PI[:,:,2].flatten(), Scaler, verbose=False)
print(f"PI coverage: {coverage*100:.4f}%, PI avg. length: {avg_length:.4f}, PI avg. length transformed: {avg_length_trans:.4f}, CWC: {cwc:.4f}")

coverage_conf, avg_length_conf, avg_length_trans_normed, cwc_conf = utils.compute_coverage_len(test_y.flatten(), PI_conf[:,:,0].flatten(), PI_conf[:,:,2].flatten(), Scaler, verbose=False)
print(f"PI conformalized coverage: {coverage_conf*100:.4f}%, PI avg. length: {avg_length_conf:.4f}, PI avg. length transformed: {avg_length_trans_normed:.4f}, CWC: {cwc_conf:.4f}")

utils.plot_PIs(test_y, PI[:,:,1],
                PI[:,:,0], PI[:,:,2],
                PI_conf[:,:,0], PI_conf[:,:,2],
                scaler=Scaler, title=f'Single CQR - {P["model_type"]}')