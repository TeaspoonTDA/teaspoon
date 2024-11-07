def random_feature_map_model(u_obs, Dr, w=0.005, b=4.0, beta=4e-5, seed=None):
    """
    Function for generating a random feature map dynamical system model based on the method presented in https://doi.org/10.1016/j.physd.2021.132911. 

    Args:
        u_obs (array): Array of observations (D x N) D is the dimension, N is the number of time points (INCLUDING TRAINING DATA)
        Dr (int): Reservoir dimension
        w (float): Random feature weight matrix distribution width parameter.
        b (float): Random feature bias vector distribution parameter.
        beta (float): Ridge regression regularization parameter.
        seed (int): Random seed (optional)
        
    Returns:
        W_LR (array): Optimal model weights.
        W_in (array): Random weight matrix.
        b_in (array): Random bias vector.
    """
    import numpy as np
    import time
    N = np.shape(u_obs)[1]
    D = np.shape(u_obs)[0]
    
    if seed:
        np.random.seed(seed=seed)
    else:
        np.random.seed(seed=int(time.time()))
    # Fix internal weights and bias
    W_in = np.random.uniform(-w,w,size=(Dr,D))
    b_in = np.random.uniform(-b,b,size=(Dr,1))

    phi_mat = np.array(np.tanh(W_in@u_obs[:,0].reshape(-1,1)+b_in)) #np.tanh(W_in@u_obs + b_in)

    for i in range(1,N):
        phi_mat = np.hstack((phi_mat, np.tanh(W_in@u_obs[:,i-1].reshape(-1,1)+b_in)))
    
    # Compute W_LR (u_obs does not include u_0)
    W_LR = u_obs@phi_mat.T@np.linalg.inv((phi_mat@phi_mat.T+beta*np.eye(Dr)))

    return W_LR, W_in, b_in


def get_forecast(X_start, W, W_in, b_in, forecast_len, auto_diff=False):
    """
    Function for computing a forecast from a given random feature map model. 

    Args:
        X_start (array): Starting point for the forecast (D x 1) vector.
        W (array): Matrix of model coefficients to use for forecasting.
        W_in (array): Random weight matrix.
        b_in (array): Random bias vector.
        forecast_len (int): Number of points to forecast into the future.
        auto_diff (bool): Toggle automatic differentiation for tensorflow usage with TADA.
        
    Returns:
        forecast (array): Array of forecasted states.
    """
    import numpy as np
    D = np.shape(X_start)[0]
    Dr = np.shape(W)[1]
    if not auto_diff:
        next_pt = X_start.reshape(-1,1)
        phi_mat = np.tanh(W_in@X_start.reshape(-1,1)+b_in)

        for i in range(1, forecast_len):
            next_pt = W@phi_mat[:,i-1].reshape(-1,1)
            phi_mat = np.hstack((phi_mat, np.tanh(W_in@next_pt+b_in)))
        forecast = np.hstack((X_start.reshape(-1,1), W@phi_mat))
    else:
        try:
            import tensorflow as tf
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
        except:
            raise ImportError("TADA Requires tensorflow for optimization")
        
        start_arr = np.zeros(shape=(np.shape(X_start)[0],(forecast_len)))
        start_arr[:,0] = X_start.reshape(-1)

        forecast = tf.constant(start_arr.T) 
        for i in range(1, forecast_len):
            index = [[i]]
            forecast = tf.tensor_scatter_nd_update(forecast, index, tf.transpose(tf.matmul(W,tf.tanh(tf.matmul(W_in, tf.reshape(forecast[i-1,:],[-1,1]))+b_in))))
    
    return forecast


def forecast_time(X_model_a, X_truth, dt=1.0, lambda_max=1.0, threshold=0.05):
    '''
        Function to compute the forecast time using the relative forecast error to compare predictions to measurements with a threshold.  

        Args:
            X_model_a (array): Array of forecast points
            X_truth (array): Array of measurements (ground truth)
            dt (float): Time step size (defaults to 1 to return number of points)
            lambda_max (float): Maximum lyapunov exponent (defaults to 1 to return number of points)
            threshold (float): Threshold to use for comparing forecast and measurements. 
        
        Returns:
            (float): Forecast time for the given threshold. 
            
    '''
    import numpy as np
    for i in range(1,X_model_a.shape[1]):
        error = np.divide(np.linalg.norm(X_model_a[:,0:i]-X_truth[:,0:i], axis=1)**2,np.linalg.norm(X_truth[:,0:i], axis=1)**2)
        if np.max(error) > threshold:
            fc_time = i*dt*lambda_max
            return fc_time
    
    raise Warning("Longer forecast required to measure forecast time.")


if __name__ == "__main__":
    import numpy as np
    from matplotlib import rc
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows import lorenz
    from teaspoon.DAF.forecasting import random_feature_map_model
    from teaspoon.DAF.forecasting import get_forecast

    # Set font
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 16})

    # Set model parameters
    Dr=300
    train_len = 4000
    forecast_len = 2000

    r_seed = 48824
    np.random.seed(r_seed)

    # Get training and tesing data at random initial condition
    ICs = list(np.random.normal(size=(3,1)).reshape(-1,))
    t, ts = lorenz(L=500, fs=50, SampleSize=6001, parameters=[28,10.0,8.0/3.0],InitialConditions=ICs)
    ts = np.array(ts) 

    # Add noise to signals
    noise = np.random.normal(scale=0.01, size=np.shape(ts[:,0:train_len+forecast_len]))
    u_obs = ts[:,0:train_len+forecast_len] + noise

    # Train model
    W_LR, W_in, b_in = random_feature_map_model(u_obs[:,0:train_len],Dr, seed=r_seed)

    # Generate forecast
    forecast_len = 500
    X_model= get_forecast(u_obs[:,train_len], W_LR, W_in, b_in,forecast_len=forecast_len)
    X_meas = u_obs[:,train_len:train_len+forecast_len]

    # Plot measurements and forecast
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X_model[0,:],'r', label="Forecast")  
    ax1.plot(X_meas[0,:], '.b', label="Measurement")
    ax1.plot([],[])
    ax1.set_title('x', fontsize='x-large')
    ax1.tick_params(axis='both', which='major', labelsize='x-large')
    ax1.set_ylim((-30,30))


    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(X_model[1,:],'r', label="Forecast")  
    ax2.plot(X_meas[1,:], '.b', label="Measurement")
    ax2.plot([],[])
    ax2.set_title('x', fontsize='x-large')
    ax2.tick_params(axis='both', which='major', labelsize='x-large')
    ax2.set_ylim((-30,30))

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(X_model[2,:],'r', label="Forecast")  
    ax3.plot(X_meas[2,:], '.b', label="Measurement")
    ax3.plot([],[])
    ax3.legend(fontsize='large', loc='upper left')
    ax3.set_title('x', fontsize='x-large')
    ax3.tick_params(axis='both', which='major', labelsize='x-large')
    ax3.set_ylim((0,60))

    plt.tight_layout()
    plt.show()

