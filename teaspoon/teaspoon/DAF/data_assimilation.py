import numpy                 as np
import gudhi                 as gd
from gudhi.tensorflow        import RipsLayer
from gudhi.wasserstein       import wasserstein_distance
from teaspoon.DAF.forecasting import get_forecast
from teaspoon.DAF.forecasting import forecast_time
from teaspoon.DAF.forecasting import random_feature_map_model

try:
    import tensorflow as tf
    from tensorflow import keras
except:
    raise ImportError("TADA Requires tensorflow for optimization")

def TADA(u_obs, window_size, model_parameters, n_epochs=1, train_len=4000, opt_params=[1e-5, 0.99], window_number=1, j2_sample_size=50):
    '''
        Compute optimal model weights using data assimilation and persistent homology. Heavily modified code from https://github.com/GUDHI/TDA-tutorial.git

        Args:
            u_obs (array): Array of observations (D x N) D is the dimension, N is the current number of time points (INCLUDING TRAINING DATA)
            window_size (int): Number of points included in the sliding window
            model_parameters (list): List of parameters used to generate a forecast. Must contain current model weights as a list tensorflow variables (W_A), original model weights (W0), list of model specific internal parameters (mu), forecast model function (G) and number of past points to use for predictions (p).
            n_epochs (int): Number of optimization epochs for each assimilation window
            train_len (int): Number of points used for training the original model
            opt_params (list): List of parameters used for gradient descent optimization. Must contain a learning rate and decay rate. Decay only occurs between assimilation windows. 
            window_number (int): Current window number. Used to determine how many points to forecast into the future for the current window.
            j2_sample_size (int): Number of points to sample for J_2 loss function. This is a random sample of the training data.
        
        Returns:
            (array): Updated model weights using TADA algorithm.
    '''

    # Grow the window size as new measurements are received
    if window_number < window_size:
        current_window_size = window_number + 1
    else:
        current_window_size = window_size

    # Unpack forecast weights from random feature maps and linear regression
    if len(model_parameters) != 5:
        raise ValueError("model_parameters must contain four items. W_A, W_LR, mu, G, p")

    W_A, W0, mu, G, p = model_parameters
    W0 = [tf.constant(w, dtype=tf.float32) for w in W0]

    start = train_len
    end = train_len + window_number + 1

    X_meas = u_obs[:,start:end][:,-current_window_size:]
    X_model = get_forecast(u_obs[:,train_len-p+1:train_len+1], W_A, mu,forecast_len=end-train_len, G=G, auto_diff=True)[:,start:end]

    # Create tensorflow variable for weight matrix initialized to original LR weights
    # W = tf.Variable(initial_value=W_A, trainable=True, name="W", dtype=tf.float64)
    W = W_A#[tf.Variable(w, dtype=tf.float32, trainable=True) for w in W_A]
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
            model_parameters = [W, W0, mu, G]
        
        start_pt = u_obs[:,train_len-p+1:train_len+1]

        # Initialize gradient tape
        with tf.GradientTape() as tape:
            # Generate forecast with current model
            forecast = get_forecast(start_pt, W, mu,forecast_len=window_number, G=G, auto_diff=True).T
        
            # Get persistence diagrams from current window forecast
            dgm = rips_layer.call(tf.cast(forecast[-current_window_size:,:],dtype=np.float32))
            dgm0 = tf.cast(dgm[0][0], dtype=np.float64)
            dgm1 = tf.cast(dgm[1][0], dtype=np.float64)

            # Compute Wasserstein distances for current window
            distance0 = wasserstein_distance(dgm0, target_pd0, order=1., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            distance1 = wasserstein_distance(dgm1, target_pd1, order=1., internal_p=2., enable_autodiff=True, keep_essential_parts=False)

            # Compute error Wasserstein distances
            reg_dgm = rips_layer.call(tf.cast(forecast[-current_window_size:,:],dtype=np.float32)-tf.constant(X_meas.T,dtype=np.float32))
            reg_dgm0 = tf.cast(reg_dgm[0][0], dtype=np.float64)
            reg_dgm1 = tf.cast(reg_dgm[1][0], dtype=np.float64)
            empty_dgm = tf.constant([], dtype=np.float64)
            reg_distance0 = wasserstein_distance(reg_dgm0, empty_dgm, order=1., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            reg_distance1 = wasserstein_distance(reg_dgm1, empty_dgm, order=1., internal_p=2., enable_autodiff=True, keep_essential_parts=False)
            
            # Loss function terms
            persistence_loss = distance1 + distance0
            reg_loss = reg_distance0 + reg_distance1
             
            # Total loss
            J_1_loss =  persistence_loss + reg_loss

            # Compute J_2 loss
            random_indices = tf.random.shuffle(tf.range(tf.shape(u_obs[:,:train_len])[0]))[:j2_sample_size]
            training_samples = u_obs[:,:train_len][:,random_indices]
            train_loss = G(training_samples, W, mu, auto_diff=True) - training_samples
            train_dgm = rips_layer.call(tf.cast(tf.squeeze(tf.gather(train_loss, random_indices)), np.float32))
            train_dgm0 = tf.cast(train_dgm[0][0], dtype=np.float64)
            train_dgm1 = tf.cast(train_dgm[1][0], dtype=np.float64)
            empty_dgm = tf.constant(np.array([]),shape=(0,2), dtype=np.float64)
            J_2_loss = wasserstein_distance(train_dgm0, empty_dgm, order=1., internal_p=2., enable_autodiff=True, keep_essential_parts=False) + wasserstein_distance(train_dgm1, empty_dgm, order=1., internal_p=2., enable_autodiff=True, keep_essential_parts=False)

            # Compute total loss
            loss = J_1_loss + J_2_loss
        
        # Compute gradient of loss function with respect to model weights
        gradients = tape.gradient(loss, W)

        # Apply gradients
        optimizer.apply_gradients(zip(gradients, W))
        
        # Store the forecast as the new model
        X_model = tf.transpose(forecast[-current_window_size:,:]).numpy()

    return W


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



def rafda(u_obs, Dr, Gamma, M=1000, g=1000, w=0.005, b=4.0, beta=4e-5, seed=None):
    """
    Function for generating a RAFDA model based on the method presented in https://doi.org/10.1016/j.physd.2021.132911. 

    Args:
        u_obs (array): Array of observations (D x N) D is the dimension, N is the number of training points.
        Dr (int): Reservoir dimension
        Gamma (array): Observational covariance matrix.
        M (int): Ensemble size.
        g (float): Initial random weight ensemble distribution parameter.
        w (float): Random feature weight matrix distribution width parameter.
        b (float): Random feature bias vector distribution parameter.
        beta (float): Ridge regression regularization parameter.
        seed (int): Random seed (optional)
        
    Returns:
        W_RAFDA (array): Optimal RAFDA model weights.
        W_in (array): Random weight matrix.
        b_in (array): Random bias vector.
    """
    D = np.shape(u_obs)[0]
    N = np.shape(u_obs)[1]
    W_LR, W_in, b_in = random_feature_map_model(u_obs, Dr, w=w, b=b, beta=beta, seed=seed)
    W_LR = W_LR.ravel()

    # Sample initial ensemble (u_obs \in R^D)
    u_o = np.random.multivariate_normal(u_obs[:,0], Gamma,size=M).T
    w_o = np.random.multivariate_normal(W_LR, g*np.identity(D*Dr),size=M).T
    Z_a = np.vstack([u_o,w_o])
    H = np.hstack([np.identity(D),np.zeros((D,D*Dr))])
    Z_f = np.zeros((D+D*Dr, M))

    for n in range(1, N):
        # Ensemble steps
        phi = np.tanh(W_in @ Z_a[:D,:] + b_in)
        W_a_prev = Z_a[D:,:].reshape((D, Dr, M))

        W_a_flat = W_a_prev.reshape(W_a_prev.shape[0] * W_a_prev.shape[1], M)
        u_f = np.einsum('ijk,jk->ik', W_a_prev, phi)

        Z_f = np.vstack((u_f, W_a_flat))
        Z_f_mean = np.mean(Z_f, axis=1, keepdims=True)
        Z_f_hat = Z_f - Z_f_mean

        P_f = (1/(M-1))*Z_f_hat@Z_f_hat.T
        Uo = np.tile(u_obs[:,n][:, np.newaxis],M) - np.sqrt(Gamma)@np.random.normal(size=(D,M))

        Z_a = Z_f - P_f@H.T@np.linalg.inv(H@P_f@H.T+Gamma)@(H@Z_f-Uo)
        Z_f = np.copy(Z_a)

    W_RAFDA = np.mean(Z_a[D:, :], axis=1).reshape((D,Dr))

    return W_RAFDA, W_in, b_in






    