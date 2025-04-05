

Data Assimilation
=========================================================

This page gives a summary of the functions available in the data assimilation library. Differentiation of persistence diagrams is exploited to optimize data driven model coefficients by minimizing topological differences between model the model forecast and measurements. More information on the details of the TADA algorithm can be found in,  "`Topological Approach for Data Assimilation <https://arxiv.org/abs/2411.18627>`_." We plan to implement more data assimilation tools here in the future. 

.. warning:: 
    `TADA` requires `tensorflow <https://www.tensorflow.org>`_ for optimization features. Please install teaspoon using the command: `pip install "teaspoon[full]"` to install the necessary packages. 
    
.. automodule:: teaspoon.DAF.data_assimilation
    :members: 


**Random Feature Map Example**::

    import numpy as np
    from teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows import lorenz
    from teaspoon.DAF.data_assimilation import TADA
    from teaspoon.DAF.forecasting import forecast_time
    from teaspoon.DAF.forecasting import random_feature_map_model
    from teaspoon.DAF.forecasting import get_forecast
    from teaspoon.DAF.forecasting import G_rfm
    import tensorflow as tf

    # Set random seed
    r_seed = 48824
    np.random.seed(r_seed)

    # Set TADA parameters
    snr = 100.0
    lr = 1e-6
    train_len = 4000
    forecast_len = 2000
    window_size = 50
    max_window_number = 100

    # Get training and validation data at random initial condition
    ics = np.random.uniform(0, 1, size=(3,))
    t, ts = lorenz(L=500, fs=50, SampleSize=6001, parameters=[28,10.0,8.0/3.0], InitialConditions=ics)
    ts = np.array(ts) 

    # Get signal and noise amplitudes using signal-to-noise ratio
    a_sig = np.sqrt(np.mean(np.square(ts),axis=1))
    a_noise = a_sig * 10**(-snr/20)

    # Add noise to the signal
    Gamma = np.diag(a_noise)
    noise = np.random.normal(size=np.shape(ts[:,0:train_len+forecast_len]))
    u_obs = ts[:,0:train_len+forecast_len] + Gamma@noise

    # Train model
    W_LR, W_in, b_in = random_feature_map_model(u_obs[:,0:train_len],Dr=300, seed=r_seed)

    # Set optimization parameters
    d_rate = 0.99
    opt_params = [lr, d_rate]
    W_opt = [tf.Variable(W_LR, trainable=True, dtype=tf.float64)]
    mu = (W_in, b_in)
    p=1

    # TADA optimization loop
    for window_number in range(1,max_window_number):
        model_parameters = [W_opt, W_LR, mu, G_rfm, p]  
        W_opt = TADA(u_obs, window_size, model_parameters, train_len=train_len, n_epochs=1, opt_params=opt_params, window_number=window_number, j2_sample_size=100)

    # Set forecast parameters
    window_number = 1000
    start = train_len
    end = train_len + window_number + 1

    # Forecast TADA and LR models and get measurements
    X_model_tada = get_forecast(u_obs[:,train_len].reshape(-1,1), W_opt, mu,forecast_len=end-train_len, G=G_rfm)
    X_model_lr =get_forecast(u_obs[:,train_len].reshape(-1,1), W_LR, mu,forecast_len=end-train_len, G=G_rfm)
    X_meas = u_obs[:,start:end]

    # Compute and print forecast times
    tada_time = forecast_time(X_model_tada, X_meas, dt=0.02, lambda_max=0.91, threshold=0.05)
    lr_time = forecast_time(X_model_lr, X_meas, dt=0.02, lambda_max=0.91, threshold=0.05)

    print(f"TADA Forecast Time: {tada_time}")
    print(f"LR Forecast Time: {lr_time}")


**LSTM Example**::

    import numpy as np
    from teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows import lorenz
    from teaspoon.DAF.forecasting import lstm_model
    from teaspoon.DAF.forecasting import get_forecast
    from teaspoon.DAF.forecasting import G_lstm
    from sklearn.preprocessing import StandardScaler
    from teaspoon.DAF.data_assimilation import TADA
    from teaspoon.DAF.forecasting import forecast_time
    import tensorflow as tf

    r_seed = 48824
    np.random.seed(r_seed)

    # Set TADA parameters
    snr = 60.0
    lr = 1e-5
    train_len = 4000
    forecast_len = 2000
    window_size = 50
    max_window_number = 10
    lr_time = 0
    tada_time = 0

    # Get training and tesing data at random initial condition
    ICs = list(np.random.normal(size=(3,1)).reshape(-1,))
    t, ts = lorenz(L=500, fs=50, SampleSize=6001, parameters=[28,10.0,8.0/3.0],InitialConditions=ICs)
    ts = np.array(ts)

    # Add noise to signals
    noise = np.random.normal(scale=0.01, size=np.shape(ts[:,0:train_len+forecast_len]))
    u_obs = ts[:,0:train_len+forecast_len] + noise

    # Scale u_obs using StandardScaler
    scaler = StandardScaler()
    u_obs = scaler.fit_transform(u_obs.T).T

    # Train model
    lstm_units = 500
    p=5
    model = lstm_model(u_obs[:,0:train_len], p=p, epochs=5, units=lstm_units, batch_size=50)

    # Set optimization parameters
    d_rate = 0.99
    opt_params = [lr, d_rate]
    mu = (lstm_units, p, model)
    W_lstm = model.get_weights()
    W_opt = model.trainable_weights

    # Set forecast parameters
    window_number = 200
    start = train_len
    end = train_len + window_number

    # Get initial forecast
    X_model_initial = get_forecast(u_obs[:,train_len-p+1:train_len+1], W_lstm, mu=(lstm_units, p, model),forecast_len=end-train_len, G=G_lstm)


    # TADA optimization loop
    for window_number in range(1,max_window_number):
        mu = (lstm_units, p, model)
        model_parameters = [W_opt, W_lstm, mu, G_lstm, p]  
        W_opt = TADA(u_obs, window_size, model_parameters, train_len=train_len, n_epochs=1, opt_params=opt_params, window_number=window_number, j2_sample_size=100)


    # Forecast original TADA model and get measurements
    X_model_tada = get_forecast(u_obs[:,train_len-p+1:train_len+1], W_opt, mu=(lstm_units, p, model),forecast_len=end-train_len, G=G_lstm)
    X_meas = u_obs[:,start:end]

    # Invert the scale on X_model and X_meas
    X_model_tada = scaler.inverse_transform(X_model_tada.T).T
    X_model_initial = scaler.inverse_transform(X_model_initial.T).T
    X_meas = scaler.inverse_transform(X_meas.T).T

    # Compute and print forecast times
    tada_time = forecast_time(X_model_tada, X_meas, dt=0.02, lambda_max=0.91, threshold=0.05)
    initial_time = forecast_time(X_model_initial, X_meas, dt=0.02, lambda_max=0.91, threshold=0.05)

    print(f"TADA Forecast Time: {tada_time}")
    print(f"Initial Forecast Time: {initial_time}")

.. note:: 
    Resulting forecast times may vary depending on the operating system.  