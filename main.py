import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from models import regression_model
import data_preprocessing
from conformal_prediction import EnCQR
import utils
import data_loaders
import pandas as pd
import time
from cp_single import CQR

B = [1]                       # number of ensembles
alpha = 0.05                # confidence level            
quantiles = [alpha/2,       # quantiles to predict
             0.5,
             1-(alpha/2)] 

# rf hyperparams
n_trees = 20                # number of trees in each rf model

# lstm tcn only
regression = 'quantile'     # options: {'quantile', 'linear'}. If 'linear', just set one quantile
l2_lambda = [1e-4]            # weight of l2 regularization in the lstm and tcn models
batch_size = [16,32]             # size of batches using to train the lstm and tcn models

# lstm hyperparams
units = [32,64]                 # number of units in each lstm layer
n_layers = [1]               # number of lstm layers in the model

# units = 64
# n_layers = 1


# tcn hyperparams
dilations = [[1,2],[1,2,4]]       # dilation rate of the Conv1D layers
n_filters = [256]             # filters in each Conv1D layer 
kernel_size = [3,7]             # kernel size in each ConvID layer
train_df, val_df, test_df = data_loaders.get_gas_data([0.8,0.2],1,"old",165)
nof_runs = 1

colnames = ['Nof Learners', 'alpha', 'batch_size', 'kernel_size', 'dilations','n_filters', 'l2_lambda', 'learning_rate','epochs', 'time_steps_in', 'coverage', 'avg_width','avg_width_transformed', 'cwc','coverage_conf', 'avg_width_conf', 'avg_width_transformed_conf', 'cwc_conf']# ['Nof Learners', 'alpha', 'batch_size', 'units', 'n_layers', 'l2_lambda', 'learning_rate','epochs', 'time_steps_in', 'coverage', 'avg_width','avg_width_transformed', 'cwc','coverage_conf', 'avg_width_conf', 'avg_width_transformed_conf', 'cwc_conf']
result_df = None
csv_path = './results_cqr_tcn.csv'

learning_rate = [1e-3]
time_steps_in = [6,12]

total_time = 0
total_runs = len(B) * len(l2_lambda) * len(batch_size) * len(units) * len(n_layers) * len(learning_rate) * len(time_steps_in) * nof_runs


for idx_ensembles in range(len(B)):
    for idx_l2 in range(len(l2_lambda)):
        for idx_batch in range(len(batch_size)):
            for idx_kernel in range(len(kernel_size)): #for idx_units in range(len(units)):
                for idx_dilations in range(len(dilations)): #for idx_layers in range(len(n_layers)):
                    for idx_filters in range(len(n_filters)):
                        for idx_learning in range(len(learning_rate)):
                            for idx_time_steps_in in range(len(time_steps_in)):
                                
                                if os.path.exists(csv_path):
                                    result_df = pd.read_csv(csv_path, sep=";", header=0)
                                else:
                                    result_df = pd.DataFrame(columns=colnames)

                                coverage_sum = 0
                                avg_length_sum = 0 
                                cwc_sum = 0
                                coverage_conf_sum = 0
                                avg_length_conf_sum = 0 
                                cwc_conf_sum = 0
                                epochs = 0
                                # Store the configuration in a dictionary
                                P = {'B':B[idx_ensembles], 'alpha':alpha, 'quantiles':quantiles,
                                    'n_trees':n_trees,  
                                    'regression':regression,'l2':l2_lambda[idx_l2], 'batch_size':batch_size[idx_batch],
                                    # 'units':units[idx_units],'n_layers':n_layers[idx_layers],
                                    'dilations':dilations[idx_dilations], 'n_filters': n_filters[idx_filters], 'kernel_size': kernel_size[idx_kernel],
                                    'learning_rate' : learning_rate[idx_learning], 'time_steps_in' : time_steps_in[idx_time_steps_in]}
                                print("Parameters: ", P)
                                
                                for run in range(nof_runs):
                                    tic = time.perf_counter()
                                    print("RUN No.: ", run)
                                

                                    # Split data, rest is test

                                    amount_train = 0.90
                                    amount_val = 0.10
                                    # amount_train = 0.7
                                    # amount_val = 0.2


                                    #train_df, val_df, test_df = data_loaders.get_solar_data()
                                    # train_df, val_df, test_df = data_loaders.get_gas_data([amount_train,amount_val],1)
                                    
                                    print(test_df.shape)

                                    train_data, val_x, val_y, test_x, test_y, Scaler = data_preprocessing.data_windowing(df=train_df, 
                                                                                                                        val_data=val_df,
                                                                                                                        test_data=test_df,
                                                                                                                        B=P['B'], 
                                                                                                                        time_steps_in=P['time_steps_in'], 
                                                                                                                        time_steps_out=1, 
                                                                                                                        label_columns=['Price'])

                                    # print("-- Training data --")
                                    # for i in range(len(train_data)):
                                    #     print(f"Set {i} - x: {train_data[i][0].shape}, y: {train_data[i][1].shape}")
                                    # print("-- Validation data --")
                                    # print(f"x: {val_x.shape}, y: {val_y.shape}")
                                    # print("-- Test data --")
                                    # print(f"x: {test_x.shape}, y: {test_y.shape}")

                                    # Update configuration dict
                                    P['time_steps_in'] = test_x.shape[1]
                                    P['n_vars'] = test_x.shape[2] 
                                    P['time_steps_out'] = test_y.shape[1]

                                    # P['model_type'] = 'tcn' 

                                    # # Train 
                                    # model = regression_model(P)
                                    # hist = model.fit(train_data[0][0], train_data[0][1], val_x, val_y, epochs=100, patience=7)
                                    # utils.plot_history(hist)

                                    # # Test
                                    # PI = model.transform(test_x)
                                    # utils.plot_PIs(test_y, PI[:,:,1],
                                    #                 PI[:,:,0], PI[:,:,2],
                                    #                 scaler=Scaler, title='TCN model') #x_lims=[0,168]

                                    # P['model_type'] = 'lstm'

                                    # # Train
                                    # model = regression_model(P)
                                    # hist = model.fit(train_data[0][0], train_data[0][1], val_x, val_y,epochs=20, patience=10, start_from_epoch=1)
                                    # utils.plot_history(hist, P['alpha'])

                                    # # Test
                                    # PI = model.transform(test_x)
                                    # utils.plot_PIs(test_y, PI[:,:,1],
                                    #                 PI[:,:,0], PI[:,:,2],
                                    #                 scaler=Scaler, title='LSTM model') #x_lims=[0,168],

                                    P['model_type'] = 'tcn'

                                    # compute the conformalized PI with EnCQR
                                    # PI, conf_PI, history = EnCQR(train_data, val_x, val_y, test_x, test_y, P)
                                    PI, conf_PI, history = CQR(train_data,val_x,val_y,test_x, test_y, P)
                                    # # Plot original and conformalized PI
                                    # utils.plot_PIs(test_y, PI[:,:,1],
                                    #             PI[:,:,0], PI[:,:,2],
                                    #             conf_PI[:,:,0], conf_PI[:,:,2],
                                    #             x_lims=[0,168], scaler=Scaler, title='EnCQR')

                                    # Compute PI coverage and length before and after conformalization
                                    print("Before conformalization:")
                                    coverage, avg_length, avg_length_transformed, cwc = utils.compute_coverage_len(test_y.flatten(), PI[:,:,0].flatten(), PI[:,:,2].flatten(), Scaler, verbose=False)
                                    print("After conformalization:")
                                    coverage_conf, avg_length_conf, avg_length_transformed_conf, cwc_conf = utils.compute_coverage_len(test_y.flatten(), conf_PI[:,:,0].flatten(), conf_PI[:,:,2].flatten(), Scaler, verbose=False)

                                    coverage_sum += coverage
                                    avg_length_sum += avg_length 
                                    cwc_sum += cwc

                                    coverage_conf_sum += coverage_conf
                                    avg_length_conf_sum += avg_length_conf 
                                    cwc_conf_sum += cwc_conf

                                    epochs += len(history.history['loss'])

                                    toc = time.perf_counter()
                                    time_elapsed = toc - tic
                                    total_time += time_elapsed
                                    total_runs -= 1
                                    print(f"<<<<<<< Run took {time_elapsed:0.4f} seconds")
                                    print(f"<<<<<<< Runs left: {total_runs}")
                                
                                result_df.loc[len(result_df)] = {'Nof Learners' : B[idx_ensembles], 'alpha' : alpha, 'batch_size' : batch_size[idx_batch], 'kernel_size' : kernel_size[idx_kernel], 'dilations': dilations[idx_dilations],"n_filters" : n_filters[idx_filters], 'l2_lambda' : l2_lambda[idx_l2],
                                                'learning_rate' : learning_rate[idx_learning], 'epochs' : (epochs / nof_runs ), 'time_steps_in' : time_steps_in[idx_time_steps_in] ,'coverage' : coverage_sum / nof_runs, 'avg_width' : avg_length_sum / nof_runs, 'avg_width_transformed' : avg_length_transformed, 
                                                'cwc' : cwc_sum / nof_runs, 'coverage_conf' : coverage_conf_sum / nof_runs, 'avg_width_conf' : avg_length_conf_sum / nof_runs, 'avg_width_transformed_conf' : avg_length_transformed_conf / nof_runs, 'cwc_conf' : cwc_conf_sum / nof_runs}
                                # result_df.loc[len(result_df)] = {'Nof Learners' : B[idx_ensembles], 'alpha' : alpha, 'batch_size' : batch_size[idx_batch], 'units' : units[idx_units], 'n_layers' : n_layers[idx_layers], 'l2_lambda' : l2_lambda[idx_l2],
                                #                 'learning_rate' : learning_rate[idx_learning], 'epochs' : (epochs / nof_runs ), 'time_steps_in' : time_steps_in[idx_time_steps_in] ,'coverage' : coverage_sum / nof_runs, 'avg_width' : avg_length_sum / nof_runs, 'avg_width_transformed' : avg_length_transformed, 
                                #                 'cwc' : cwc_sum / nof_runs, 'coverage_conf' : coverage_conf_sum / nof_runs, 'avg_width_conf' : avg_length_conf_sum / nof_runs, 'avg_width_transformed_conf' : avg_length_transformed_conf / nof_runs, 'cwc_conf' : cwc_conf_sum / nof_runs}
                                result_df.to_csv(csv_path,sep=';', index=False)
                        

print(f"<<<<<<< Experiment took {total_time:0.4f} seconds")
