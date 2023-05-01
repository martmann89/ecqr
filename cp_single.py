import numpy as np
from models import regression_model
import utils

def CQR(train_data, val_x, val_y, test_x, test_y, P, plot_training=False):
    P_score = {'regression' : 'linear', 'model_type' : 'lstm', 'n_layers' : 1, 'units' : 32, 'batch_size' : P['batch_size'], 'time_steps_out' : 1, 'time_steps_in' : 6, 'quantiles' : [0.5], 'l2' : P['l2'] }
    s = P['time_steps_out']
    model = regression_model(P)
    hist = model.fit(train_data[0][0], train_data[0][1], val_x, val_y, epochs=1000, patience=50, start_from_epoch=140, learning_rate=P['learning_rate'])
    if plot_training:
        utils.plot_history(hist, P['alpha'])
    pred = model.transform(val_x)
    
    
    e_low, e_high = utils.asym_nonconformity(label=val_y, low=pred[0], high=pred[2])       # eq. (8) # on val data
    
    # model_score = regression_model(P_score)
    # model.fit(val_x, e_low)

    e_low = np.array(e_low).flatten()
    e_high = np.array(e_high).flatten()
    
    PI = model.transform(test_x)
    PI_conf = np.zeros(PI.shape)
    PI_conf[:,:,1] = PI[:,:,1]

    for i in range(test_y.shape[0]):   
        e_quantile_lo = np.quantile(e_low, 1-P['alpha'])      # 1-P['alpha']/2              # eq. (9)
        e_quantile_hi = np.quantile(e_high, 1-P['alpha'])     # 1-P['alpha']/2
        PI_conf[i,:,0] = PI[i,:,0] - e_quantile_lo                            # eq. (9)
        PI_conf[i,:,2] = PI[i,:,2] + e_quantile_hi
    
        # update epsilon with the last s steps
        e_l, e_h = utils.asym_nonconformity(label=test_y[i,:],
                                                low=PI[i,:,0],
                                                high=PI[i,:,2])
        e_low = np.delete(e_low,slice(0,s,1))
        e_high = np.delete(e_high,slice(0,s,1))
        e_low = np.append(e_low, e_l)
        e_high = np.append(e_high, e_h)
    
    return PI, PI_conf, hist
                                                   
                                                  