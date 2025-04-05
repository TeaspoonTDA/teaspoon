import unittest

class daf_module(unittest.TestCase):

    def test_tada(self):
        import numpy as np
        from teaspoon.MakeData.DynSysLib.autonomous_dissipative_flows import lorenz
        from teaspoon.DAF.data_assimilation import TADA
        from teaspoon.DAF.forecasting import forecast_time
        from teaspoon.DAF.forecasting import random_feature_map_model
        from teaspoon.DAF.forecasting import get_forecast
        from teaspoon.DAF.forecasting import G_rfm

        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow is not available")

        # Set random seed
        r_seed = 123456
        np.random.seed(r_seed)

        # Set TADA parameters
        snr = 60.0
        lr = 1e-6
        train_len = 4000
        forecast_len = 2000
        window_size = 50
        max_window_number = 30

        # Get training and validation data at random initial condition
        import os
        print(os.getcwd())
        ts = np.load('./tests/DAF_lorenz_sim_data.npy') 

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

        print("TADA Time: ", tada_time)
        print("LR Time: ", lr_time)

        self.assertAlmostEqual(tada_time, 3.894, delta=0.05)
        self.assertAlmostEqual(lr_time, 3.058, delta=0.05)



if __name__ == '__main__':
    unittest.main()







