import numpy                 as np
import gudhi                 as gd
from gudhi.tensorflow        import RipsLayer
from gudhi.wasserstein       import wasserstein_distance
from teaspoon.DAF.forecasting import get_forecast
from teaspoon.DAF.forecasting import forecast_time

try:
    import tensorflow as tf
    from tensorflow import keras
except:
    raise ImportError("TADA Requires tensorflow for optimization")

def TADA(u_obs, window_size, model_parameters, n_epochs=1, train_len=4000, opt_params=[1e-5, 0.99], window_number=1):
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
            (array): Optimal model weights using TADA algorithm.
    '''

    if window_number < window_size:
        current_window_size = window_number + 1
    else:
        current_window_size = window_size

    # Unpack forecast weights from random feature maps and linear regression
    if len(model_parameters) != 4:
        raise ValueError("model_parameters must contain four items. W_A, W_LR, W_in, b_in")

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
    
    # Decay learning rate
    if window_number > 1:
        l_rate = l_rate * decay_rate**window_number

    # Initialize optimizer
    optimizer = keras.optimizers.Adam(learning_rate=l_rate)

    # Compute initial persistence diagrams for measurement and model
    target_pd0, target_pd1 = get_initial_pds(X_meas)
    model_pd0, model_pd1 = get_initial_pds(X_model)
    dgm0, dgm1 = model_pd0, model_pd1
    
    #  Optimization Loop (allows for multiple epochs)
    for epoch in range(n_epochs):
        if epoch != 0:
            model_parameters = [W.numpy(), W_LR, W_in, b_in]
        
        start_pt = u_obs[:,train_len]

        # Initialize gradient tape
        with tf.GradientTape() as tape:
            # Generate forecast with current model
            forecast = get_forecast(start_pt, W, W_in, b_in,forecast_len=window_number+1, auto_diff=True)
        
            # Get persistence diagrams from current window forecast
            dgm = rips_layer.call(tf.cast(forecast[-current_window_size:,:],dtype=np.float32))
            dgm0 = tf.cast(dgm[0][0], dtype=np.float64)
            dgm1 = tf.cast(dgm[1][0], dtype=np.float64)

            # Compute Wasserstein distances for current window
            distance0 = wasserstein_distance(dgm0, target_pd0, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            distance1 = wasserstein_distance(dgm1, target_pd1, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)

            # Compute error Wasserstein distances
            reg_dgm = rips_layer.call(tf.cast(forecast[-current_window_size:,:],dtype=np.float32)-tf.constant(X_meas.T,dtype=np.float32))
            reg_dgm0 = tf.cast(reg_dgm[0][0], dtype=np.float64)
            reg_dgm1 = tf.cast(reg_dgm[1][0], dtype=np.float64)
            empty_dgm = tf.constant([], dtype=np.float64)
            reg_distance0 = wasserstein_distance(reg_dgm0, empty_dgm, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            reg_distance1 = wasserstein_distance(reg_dgm1, empty_dgm, order=2., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            
            # Loss function terms
            persistence_loss = distance1 + distance0
            reg_loss = reg_distance0 + reg_distance1
             
            # Total loss
            loss =  persistence_loss + reg_loss
        
        
        # Compute gradient of loss function with respect to model weights
        gradients = tape.gradient(loss, [W])

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, [W]))
        
        # Store the forecast as the new model
        X_model = tf.transpose(forecast[-current_window_size:,:]).numpy()

    return W.numpy()


def get_initial_pds(X):
    '''
        Function to compute the initial persistence diagrams of the measurements and model forecast using the Vietoris Rips complex. 

        Args:
            X (array): Point cloud array for computing persistence.
        
        Returns:
            (2 arrays): 0D and 1D persistence pairs.
    '''

    # Plot measured persistence diagram
    st = gd.RipsComplex(points=np.transpose(X)).create_simplex_tree(max_dimension=2)
    dgm = st.persistence()

    pd0 = []
    pd1 = []

    # Separate 0D and 1D persistence features
    for pair in dgm:
        if pair[0] == 0 and ~np.isinf(pair[1][1]):
            pd0.append(pair[1])
        elif pair[0] == 1:
            pd1.append(pair[1])

    pd0 = tf.convert_to_tensor(np.array(pd0))
    pd1 = tf.convert_to_tensor(np.array(pd1))
    return pd0, pd1






    