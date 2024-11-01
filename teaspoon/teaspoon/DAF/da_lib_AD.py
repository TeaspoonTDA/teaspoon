"""
    Library for topological data assimilation.

"""
import numpy                 as np

import pandas                as pd
import gudhi                 as gd
from gudhi.tensorflow        import RipsLayer
from matplotlib.gridspec     import GridSpec
import os
from gudhi.wasserstein       import wasserstein_distance
from lr_forecast import get_forecast


try:
    import tensorflow as tf
    from tensorflow import keras
except:
    raise ImportError("TADA Requires tensorflow for optimization")

def TADA(u_obs, window_size, model_parameters, n_epochs=1, train_len=4000, opt_params=[1e-8, 0.9, 1], window_number=1):
    '''
        Compute optimal model weights using data assimilation and persistent homology. Heavily modified code from https://github.com/GUDHI/TDA-tutorial.git

        Args:
            u_obs (array): Array of observations (D x N) D is the dimension, N is the number of time points (INCLUDING TRAINING DATA)
            window_size (int): Number of points included in the sliding window
            model_parameters (list): List of parameters used to generate a forecast. Must contain current model weights (W_A), original model weights (W_LR), random feature map weight matrix (W_in) and random feature map bias vector (b_in).
            n_epochs (int): Number of optimization epochs for each assimilation window
            train_len (int): Number of points used for training the original model
            opt_params (list): List of parameters used for gradient descent optimization. Must contain a learning rate and decay rate. Decay only occurs between assimilation windows. 
            window_number (int): Current window number. Used to determine how many points to forecast into the future for the current window.
        
        Returns:
            W (array): Optimal model weights using TADA algorithm.
    '''

    if window_number < window_size:
        current_window_size = window_number + 1
    else:
        current_window_size = window_size

    # Unpack forecast weights from random feature maps and linear regression
    W_A, W_LR, W_in, b_in = model_parameters

    W_LR = tf.constant(W_LR)

    start = train_len
    end = train_len + window_number + 1

    X_meas = u_obs[:,start:end][:,-current_window_size:]
    X_model = get_forecast(u_obs[:,train_len], W_A, W_in, b_in,forecast_len=end-train_len)[:,start:end]

    # Create tensorflow variable for weight matrix initialized to original LR weights
    W = tf.Variable(initial_value=W_A, trainable=True, name="W", dtype=tf.float64)
    
    # Create rips layer for computing attaching edges 
    rips_layer = RipsLayer(homology_dimensions=[0,1])

    # Define optimizer and hyperparameters
    l_rate = opt_params[0]
    decay_rate = opt_params[1]
    
    if window_number > 1:
        l_rate = l_rate * decay_rate**window_number

    optimizer = keras.optimizers.Adam(learning_rate=l_rate)

    # Compute initial persistence diagrams for measurement and model
    target_pd0, target_pd1 = get_initial_pds(X_meas)
    model_pd0, model_pd1 = get_initial_pds(X_model)
    dgm0, dgm1 = model_pd0, model_pd1


    # Initialize lists for loss values and persistence diagrams
    losses = []
    
    #  Optimization Loop
    for epoch in range(n_epochs):
        if epoch != 0:
            model_parameters = [W.numpy(), W_LR, W_in, b_in]
        

        start_pt = u_obs[:,train_len].reshape(-1,1)

        # Optimization
        with tf.GradientTape() as tape:
            # Generate forecast with current model
            
            forecast = get_forecast(start_pt, W, W_in, b_in,forecast_len=window_number+1, auto_diff=True)
        
            # Get persistence diagrams from current window forecast
            dgm = rips_layer.call(tf.cast(forecast[-current_window_size:,:],dtype=np.float32))
            dgm0 = tf.cast(dgm[0][0], dtype=np.float64)
            dgm1 = tf.cast(dgm[1][0], dtype=np.float64)

            # Compute wasserstein distances for current window
            distance0 = wasserstein_distance(dgm0, target_pd0, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            distance1 = wasserstein_distance(dgm1, target_pd1, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)

            # Persistence Loss function
            persistence_loss = distance1 + distance0
            

            reg_dgm = rips_layer.call(tf.cast(forecast[-current_window_size:,:],dtype=np.float32)-tf.constant(X_meas.T,dtype=np.float32))
            reg_dgm0 = tf.cast(reg_dgm[0][0], dtype=np.float64)
            reg_dgm1 = tf.cast(reg_dgm[1][0], dtype=np.float64)
            empty_dgm = tf.constant([], dtype=np.float64)
            reg_distance0 = wasserstein_distance(reg_dgm0, empty_dgm, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            reg_distance1 = wasserstein_distance(reg_dgm1, empty_dgm, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            
            reg_loss = reg_distance0 + reg_distance1
             
            # Total loss
            loss =  persistence_loss + reg_loss
        
        
        # Compute gradient of loss function with respect to model weights
        gradients = tape.gradient(loss, [W])

        # Apply gradients and store losses
        optimizer.apply_gradients(zip(gradients, [W]))
        losses.append(loss.numpy())
        
        # Store the forecast as the new model
        X_model = tf.transpose(forecast[-current_window_size:,:]).numpy()


    return W.numpy()


def get_initial_pds(X):
    '''
        Function to compute the initial persistence diagrams of the measurements and model forecast using the Vietoris Rips complex. 

        Args:
            X (array): Point cloud array for computing persistence.
        
        Returns:
            pd0 (array): Array of 0D persistence pairs.
            pd1 (array): Array of 1D persistence pairs.
    '''

    # Plot measured persistence diagram
    st = gd.RipsComplex(points=np.transpose(X)).create_simplex_tree(max_dimension=2)
    dgm = st.persistence()

    pd0 = []
    pd1 = []

    for pair in dgm:
        if pair[0] == 0 and ~np.isinf(pair[1][1]):
            pd0.append(pair[1])
        elif pair[0] == 1:
            pd1.append(pair[1])

    pd0 = tf.convert_to_tensor(np.array(pd0))
    pd1 = tf.convert_to_tensor(np.array(pd1))
    return pd0, pd1



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
            fc_time (float): Forecast time for the given threshold. 
            
    '''
    for i in range(1,X_model_a.shape[1]):
        error = np.divide(np.linalg.norm(X_model_a[:,0:i]-X_truth[:,0:i], axis=1)**2,np.linalg.norm(X_truth[:,0:i], axis=1)**2)
        # print(np.max(error))
        if np.max(error) > threshold:
            fc_time = i*dt*lambda_max
            return fc_time
    
    raise Warning("Longer forecast required to measure forecast time.")


    