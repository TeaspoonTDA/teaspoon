def random_feature_map_model(u_obs, Dr, w=0.005, b=4.0, beta=4e-5, seed=None):
    """
    Function for generating a random feature map dynamical system model based on the method presented in https://doi.org/10.1016/j.physd.2021.132911. 

    Args:
        u_obs (array): Array of observations (D x N) D is the dimension, N is the number of training points.
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

    phi_mat = np.tanh(W_in @ u_obs[:,:-1] + b_in)

    # Compute W_LR (u_obs does not include u_0)
    W_LR = u_obs[:,1:]@phi_mat.T@np.linalg.inv((phi_mat@phi_mat.T+beta*np.eye(Dr)))

    return W_LR, W_in, b_in


def G_rfm(xn, W_LR, mu, auto_diff=False):
    """
    Random Feature Map Model Forecast Function with TensorFlow differentiation support.

    Args:
        xn (array): Input state vector (D x 1).
        W_LR (array): Matrix of model coefficients.
        mu (list): List of internal model parameters (W_in, b_in).
        auto_diff (bool): Toggle automatic differentiation for tensorflow usage with TADA.
    
    Returns:
        x_new (array): Forecasted state vector (D x 1).
    """
    import numpy as np
    W_in, b_in = mu
    
    if auto_diff:
        W_LR = W_LR[0]
        try:
            import tensorflow as tf
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
        except:
            raise ImportError("TensorFlow is required for auto_diff functionality.")
        x_new = tf.matmul(W_LR, tf.tanh(tf.matmul(W_in, xn) + b_in))
        
    else:
        x_new = (W_LR @ np.tanh(W_in @ xn + b_in)).reshape(-1,1)
    return x_new



def lstm_model(u_obs, units=500, p=1, epochs=50, batch_size=50):
    """
    Function for generating a LSTM forecast model 

    Args:
        u_obs (array): Array of observations (D x N) D is the dimension, N is the number of training points.
        units (int): Number of LSTM units.
        p (int): Number of past points to use for predicting the next point.
        epochs (int): Number of LSTM training epochs.
        batch_size (int): Training batch size for LSTM model.
        
    Returns:
        model (keras Sequential): Trained LSTM model.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Reshape
    import numpy as np

    D = u_obs.shape[0]
    N = u_obs.shape[1]

    # Prepare input output data for LSTM
    X_train, Y_train = [], []
    for i in range(N - p):
        X_train.append(u_obs[:,i:i+p].T)
        Y_train.append(u_obs[:,i+p].reshape(1,-1))
    
    # Reshape for LSTM (samples, time steps, features)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_train = X_train[..., np.newaxis]
    
    # Build the LSTM model
    model = Sequential([
        LSTM(units, return_sequences=False, input_shape=(p, D)),
        Dense(D),
        Reshape((1, D))
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    return model


def G_lstm(xp, W_lstm, mu, auto_diff=False):
    """
    LSTM Forecast Function with TensorFlow differentiation support.

    Args:
        xp (np.ndarray or tf.Tensor): Input state vector of shape (D, p)
        W_lstm (list): Model weight arrays to set for the LSTM model (not used here but required for compatibility with other models)
        mu (list): List containing internal model parameters (lstm_units, p, model)
        auto_diff (bool): Enable automatic differentiation functionality
    Returns:
        x_new: Forecasted state vector of shape (D, 1)
    """
    import tensorflow as tf
    import numpy as np

    _, _, model = mu

    # Ensure input is a tf.Tensor of float32
    if not isinstance(xp, tf.Tensor):
        xp = tf.convert_to_tensor(xp, dtype=tf.float32)
    else:
        xp = tf.cast(xp, dtype=tf.float32)

    # Reshape: from (D, p) to (1, p, D)
    xp_reshaped = tf.transpose(xp)  # (p, D)
    xp_reshaped = tf.expand_dims(xp_reshaped, axis=0)  # (1, p, D)

    if auto_diff:
        # Differentiable model call
        x_new = model(xp_reshaped, training=True)  # (1, D)
    else:
        # Non-differentiable prediction
        x_new = model(xp_reshaped)  # (1, D)

    x_new = tf.squeeze(x_new, axis=0)  # (D,) if shape was (1, D)
    x_new = tf.reshape(x_new, (-1, 1))  # (D, 1)
    x_new = tf.cast(x_new, dtype=tf.float64)

    return x_new



def get_forecast(Xp, W, mu, forecast_len, G=G_rfm, auto_diff=False):
    """
    Function for computing a forecast from a given forecast model. 

    Args:
        Xp (array): Starting point(s) for the forecast (D x p) vector.
        W (array/object): Model weights or model object for making predictions.
        mu (list): List of internal model parameters (model specific).
        forecast_len (int): Number of points to forecast into the future.
        G (function): Forecast function. Defaults to random feature map (G_rfm).
        auto_diff (bool): Toggle automatic differentiation for tensorflow usage with TADA.
        
    Returns:
        forecast (array): Array of forecasted states.
    """
    import numpy as np
    D = np.shape(Xp)[0]
    p = np.shape(Xp)[1]

    if not auto_diff:
        x_next = G(Xp, W, mu)
        forecast = Xp[:,-p:].reshape(D,p)
        for i in range(1, forecast_len):
            x_next = G(forecast[:,-p:].reshape(D,p), W, mu)
            forecast = np.hstack((forecast, x_next))
    else:
        try:
            import tensorflow as tf
            from tensorflow.python.ops.numpy_ops import np_config
            np_config.enable_numpy_behavior()
        except ImportError:
            raise ImportError("TensorFlow is required for auto_diff functionality.")
        
        forecast = tf.reshape(tf.convert_to_tensor(Xp[:,-p:], dtype=tf.float64), (D,p))
        for i in range(forecast_len):
            x_next = G(forecast[:,-p:].reshape(D,p), W, mu, auto_diff=True)
            forecast = tf.concat([forecast, x_next], axis=1)
    
    return forecast[:,p-1:]


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
    X_model= get_forecast(u_obs[:,train_len].reshape(-1,1), W_LR, mu=(W_in, b_in),forecast_len=forecast_len)
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

