
def UserGuide(list_of_autonomous_dissipative_flows,
              list_of_conservative_flows,
              list_of_periodic_functions,
              list_of_maps,
              list_of_noise_models,
              list_of_medical_data,
              list_of_delayed_flows,
              list_of_driven_dissipative_flows):
        print('')
        print('----------------------------------------User Guide----------------------------------------------')
        print('')
        print('This code outputs a time array t and a list time series for each variable of the dynamic system.')
        
        print('The user is only required to enter the system (see list below) as a string and ')
        print('the dynamic state as either periodic or chaotic as a string.')
        print('')
        print('The user also has the optional inputs as the time series length in seconds (L), ')
        print('the sampling rate (fs), and the sample size (SampleSize).')
        print('If the user does not supply these values, they are defaulted to preset values.')
        print('')
        print('Other optional inputs are parameters and InitialConditions. The parameters variable') 
        print('needs to be entered as a list or array and are the dynamic system parameters.')
        print('If the correct number of parameters is not provided it will default to preset parameters.')
        print('The InitialConditions variable is also a list or array and is the initial conditions of the system.')
        print('The length of the initial conditions also need to match the system being analyzed.')
        print('')
        print('List of the dynamic systems available: ')
        
        
        print('___________________')
        print('Maps:')
        print('-------------------')
        
        i = 0
        for s in list_of_maps:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        print('')  
        
        
        print('___________________')
        print('Autonomous Dissipative Flows:')
        print('-------------------')
        i = 0
        for s in list_of_autonomous_dissipative_flows:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        print('') 
        
        
        print('___________________')
        print('Driven Dissipative Flows:')
        print('-------------------')
        i = 0
        for s in list_of_driven_dissipative_flows:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        print('') 
        
        
        print('___________________')
        print('Conservative Flows:')
        print('-------------------')
        
        i = 0
        for s in list_of_conservative_flows:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        print('') 
        
        
        print('___________________')
        print('Periodic Functions:')
        print('-------------------')
        
        for s in list_of_periodic_functions:
            print(s)
        print('___________________')
        print('')  
        print('') 
        
        
        print('___________________')
        print('Noise Models:')
        print('-------------------')
        
        i = 0
        for s in list_of_noise_models:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        print('') 
        
        
        print('___________________')
        print('Human Data:')
        print('-------------------')
        
        i = 0
        for s in list_of_medical_data:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        print('') 
        
        
        print('___________________')
        print('Delayed Flows:')
        print('-------------------')
        i = 0
        for s in list_of_delayed_flows:
            i = i+1
            print(i,':',s)
        print('___________________')
        print('')  
        
        print('------------------------------------------------------------------------------------------------')
