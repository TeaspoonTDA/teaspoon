#%%
import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt
from teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows import lorenz

# from forecast_layer import forecast_layer
from teaspoon.DAF.data_assimilation import TADA
from teaspoon.DAF.forecasting import forecast_time
from teaspoon.DAF.forecasting import random_feature_map_model
from teaspoon.DAF.forecasting import get_forecast
import colorednoise as cn

# Set font
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Generate Data
system = 'lorenz'
dynamic_state='chaotic'
t, ts = lorenz(L=500, fs=50, SampleSize=6001, parameters=[28,10.0,8.0/3.0],InitialConditions = [10.0**-10.0, 0.0, 1.0])
# train_len = 4000
# forecast_len = 0
# ts = np.array(ts)
# u_obs = ts


# Initialize lists and parameters

Dr=300
err_means = []
err_std = []
lr_err_means = []
lr_err_std = []
snr_list = np.linspace(100,0,10)
train_len = 4000
forecast_len = 2000
sample_freq = 50
# Sliding Window Parameters
window_size = 50
max_window_number = 10
lam_max=0.91

times = []




def model_iteration(snr, i, beta=0, M=1):
    lr = 1e-5
    train_len = 4000
    forecast_len = 2000
    sample_freq = 50
    # Sliding Window Parameters
    window_size = 50
    # max_window_number = 200
    lam_max=0.91

    r_seed = 48824
    np.random.seed(r_seed)

    # Get training and validation data at random initial condition
    ICs = list(np.random.normal(size=(3,1)).reshape(-1,))
    t, ts = lorenz(L=500, fs=sample_freq, SampleSize=6001, parameters=[28,10.0,8.0/3.0],InitialConditions=ICs)
    ts = np.array(ts) 

    a_sig = np.sqrt(np.mean(np.square(ts),axis=1))

    a_noise = a_sig * 10**(-snr/20)

    # Add noise to the signal
    Gamma = np.diag(a_noise)#eta*np.identity(np.shape(ts)[0]) 
    noise = cn.powerlaw_psd_gaussian(beta, np.shape(ts[:,0:train_len+forecast_len]), random_state=r_seed)
    u_obs = ts[:,0:train_len+forecast_len] + Gamma@noise

    # Train model
    W_LR, W_in, b_in = random_feature_map_model(u_obs[:,0:train_len],Dr, seed=r_seed)
    model_parameters = [W_LR, W_LR, W_in, b_in]

    # Set optimization parameters
    lr_0 = lr
    d_rate = 0.99
    opt_params = [lr_0, d_rate]
    W_opt = W_LR 


    for window_number in range(1,max_window_number):
        model_parameters = [W_opt, W_LR, W_in, b_in]  
        W_opt = TADA(u_obs, window_size, model_parameters, train_len=train_len, n_epochs=1, opt_params=opt_params, window_number=window_number)


    window_number = 1000
    window_size=50

    start = train_len
    end = train_len + window_number + 1


    model_parameters = [W_opt, W_LR, W_in, b_in]
    X_model_an= get_forecast(u_obs[:,train_len], W_opt, W_in, b_in,forecast_len=end-train_len)

    X_meas = u_obs[:,start:end]

    # Train new LR model using longer training set
    model_parameters = [W_LR, W_LR, W_in, b_in]
    X_model=get_forecast(u_obs[:,train_len], W_LR, W_in, b_in,forecast_len=end-train_len)


    tada_time = forecast_time(X_model_an, X_meas, dt=0.02, lambda_max=0.91, threshold=0.05)
    lr_time = forecast_time(X_model, X_meas, dt=0.02, lambda_max=0.91, threshold=0.05)
    

    return tada_time, lr_time


if model_iteration(60.0,1)[0] - 2.8574 < 0.001:
    print("PASS")
else:
    raise ValueError("Incorrect Forecast Time")

# %%
