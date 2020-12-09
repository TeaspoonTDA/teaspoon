
# In[ ]:
"""
Statistical analysis of the Fourier spectrum for time delay (tau).
=======================================================================

This function a statistical analysis (using the least median of squares) of the fourier spectrum (FFT) from a time series to detect 
the greatest signficant frequency, which is then used to select the delay (tau) from the samping rate criteria developed in 
On the 0/1 test for chaos in continuous systems by Melosik and Marszalek.
"""

def LMSforDelay(ts, fs, plotting = False):#1-D lms function
    """This function takes a time series (ts) and the sampling frequency (fs) and uses 
    the fast fourier transform and gaussian noise statistics in the fourier spectrum 
    (using least median of squares (LMS) in statistical analysis) to determine a suitable
    embedding delay for permutation entropy using the theory developed in 
    On the 0/1 test for chaos in continuous systems by Melosik and Marszalek. 
    
    Args:
       ts (array):  Time series (1d).
       fs (float): Sampling frequency of time series (equispaced).

    Kwargs:
       plotting (bool): Plotting for user interpretation. defaut is False.

    Returns:
       (int): tau, The embedding delay for permutation formation.

    """
    
    Significance_Cutoff = 0.01
    #Input: (ts) time series of data (one dimensional), (fs) sampling frequency
    #Output: Suggested delay for downsampling data set. If 0, the time series is undersampled.
    import numpy as np 
    import matplotlib.pyplot as plt
    
    xf, yf = AbsFFT(ts, fs) #gets |FFT| domain resulting from time series
    
    
    n = len(yf)
    h = int((n/2)+1) #poisition of median rounded up
    z = np.sort(yf) #Ordered Observations

    #Finding smallest difference
    z0, z1 = np.split(z,[h-1])[0] , np.split(z,[h-1])[1]#breaks x into two arrays for finding minimum difference
    l0, l1 = len(z0), len(z1)
    
    if l0 != l1: #verifies the two arrays are the same length
        z1 = z1[:-1] #removes last element from x1 to make l1 and l0 the same size
        
    z_diff = z1-z0
    z_diff_min_index = np.argmin(z_diff)
    
    #finding the best fit value ignoring outliers
    b = 0.5*z0[z_diff_min_index]+0.5*z1[z_diff_min_index]
    cutoff = Significance_Cutoff+7*b #the 5 times the best fit point (b) is from statstical analysis
                        #the +0.000001 is a 0.00001% cutoff guarentee; If there is no noise in the system, 
                        #LMS will best fit to 0.
    fmax = MaximumFrequency(xf,yf, cutoff) #gets maximum frequency from |FFT|
    delay = (fs/(3*fmax)) #Suggested delay from Melosik's paper (0/1 test)
    
    if int(delay) == 0:
        delay = 1
        
    if plotting == True:
        TextSize = 14
        plt.figure(1,figsize = (6,3))
        plt.plot(xf, yf, 'k', linewidth = 2)
        plt.plot([0,max(xf)],[cutoff,cutoff],'b', linewidth = 2, label = 'Cutoff: '+str(round(cutoff,3)))
        plt.plot(fmax,cutoff,'r.',markersize = 15, label = 'Max Freq: '+str(round(fmax,3)))
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.ylabel(r'Normalized $|\rm FFT|$', size = TextSize)
        plt.xlabel('Frequency (Hz)', size = TextSize)
        plt.grid()
        plt.legend(loc = 'upper right', fontsize = TextSize)
        plt.show()
        
    return int(delay)

def MaximumFrequency(xf,yf, cutoff):
    """This function returns the maximum frequency in fourier spectrum given a cutoff threshold.
    
    Args:
       xf (array): frequency array from FFT.
       yf (array): normalized FFT values.

    Returns:
       (float): maxfreq, the maximum frequency above threshold in fourier spectrum.

    """
    
    check = 0
    for i in range(len(yf)): #finds max frequency and number of counts with a high FFT magnitude
        if yf[i] > cutoff: #if the |FFT| is greater than 5 times the 1-D LMS fit
            maxfreq = xf[i] #finds last frequency with significance outside of cutoff
            check = 1
    if check ==0:
        maxfreq = max(xf)
        print('No signficant frequency. Possibly too much noise.')
    return maxfreq 


def AbsFFT(ts, fs): 
    """This function returns the maximum frequency in fourier spectrum given a cutoff threshold.
    
   Args:
       ts (array):  Time series (1d).
       fs (float): Sampling frequency of time series (equispaced).

    Returns:
       (array): xf and yf, x and y coordinate values for the FFT.

    """
    from scipy.fftpack import fft
    import numpy as np
    #time series must be one dimensional array
    fs = fs/2
    ts = ts.reshape(len(ts,))
    t = 1/fs #calculates the time between each data point
    N = len(ts)

    #creates array for time data points to associate to time series
    xf = np.split(np.linspace(0.0, 1.0/t, N//2),[1])[1] 
    #converts array of time data points to appropriate frequency range and removes first term

    yf = fft(ts) #computes fast fourier transform on time series
    yf = (np.abs(yf[0:N//2])) #converts fourier transform to spectral density scale
    yf = np.split(yf,[1])[1] #removes first term (tends to be infinity or overly large)
    yf = yf/max(yf) #Normalizes fourier transform based on maximum density
    return(xf, yf)



# In[ ]:


# _______________________________________EXAMPLE_________________________________________
if __name__ == "__main__":

    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100) 
    ts = np.sin(t) + np.random.normal(0,0.1, len(t))

    tau = LMSforDelay(ts, fs, plotting = True)
    print('Permutation Embedding Delay: ' + str(int(tau)))

        
    
    
