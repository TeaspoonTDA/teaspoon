def medical_data(system, dynamic_state = None, L = None, fs = None, 
         SampleSize = None, parameters = None, InitialConditions = None):
    import numpy as np
    run = True
    if run == True:
        if system == 'ECG':
            from scipy.misc import electrocardiogram
            
            if dynamic_state == 'periodic': #healthy
                ts = [electrocardiogram()[3000:5500]]
            
            if dynamic_state == 'chaotic': #heart arrythmia
                ts = [electrocardiogram()[8500:11000]]
                
            fs = 360
            ts = [(ts[0])]
            t = np.arange(len(ts[0]))/fs     
                
    # In[ ]: Complete
        
        if system == 'EEG':
            
            if SampleSize == None: SampleSize = 5000
            
            if dynamic_state == 'periodic':#healthy
                ts = [np.loadtxt('Data\\EEG\\Z093.txt',skiprows=1)[0:SampleSize]]
            
            if dynamic_state == 'chaotic':#seizure
                ts = [np.loadtxt('Data\\EEG\\S056.txt',skiprows=1)[0:SampleSize]]
                
            fs = 173.61
            t = np.arange(len(ts[0]))/fs    
            t = t[-SampleSize:]
            ts = [(ts[0])[-SampleSize:]]
            
    return t, ts