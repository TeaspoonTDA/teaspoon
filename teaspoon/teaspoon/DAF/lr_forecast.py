import tensorflow


def lr_forecast(u_obs, Dr, w=0.005, b=4.0, beta=4e-5, seed=None):
    import numpy as np
    import time
    N = np.shape(u_obs)[1]
    D = np.shape(u_obs)[0]
    # Generate phi matrix
    W_in, b_in = rand_features(D,Dr,w,b, seed=seed)

    phi_mat = np.array(np.tanh(W_in@u_obs[:,0].reshape(-1,1)+b_in)) #np.tanh(W_in@u_obs + b_in)

    for i in range(1,N):
        phi_mat = np.hstack((phi_mat, np.tanh(W_in@u_obs[:,i-1].reshape(-1,1)+b_in)))
    
    # Compute W_LR (u_obs does not include u_0)
    W_LR = u_obs@phi_mat.T@np.linalg.inv((phi_mat@phi_mat.T+beta*np.eye(Dr)))

    return W_LR, W_in, b_in


def get_forecast(X_start, W, W_in, b_in, forecast_len, auto_diff=False):
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




def rand_features(D, Dr, w, b, seed=None):
    import numpy as np
    import time
    if seed:
        np.random.seed(seed=seed)
    else:
        np.random.seed(seed=int(time.time()))
    # Fix internal weights and bias
    W_in = np.random.uniform(-w,w,size=(Dr,D))
    b_in = np.random.uniform(-b,b,size=(Dr,1))

    return W_in, b_in



if __name__ == "__main__":
    import numpy as np