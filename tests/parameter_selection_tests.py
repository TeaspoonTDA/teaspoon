# Tests done by Audun Myers as of 11/5/20 (Version 0.0.1)


# In[ ]: mutual infomration

from teaspoon.parameter_selection.MI_delay import MI_for_delay
import numpy as np

fs = 10
t = np.linspace(0, 100, fs*100)
ts = np.sin(t) + np.sin((1/np.pi)*t)

tau = MI_for_delay(ts, plotting = True, method = 'basic', h_method = 'sturge', k = 2, ranking = True)
print('Delay from MI: ',tau)



# In[ ]: auto-correlation

from teaspoon.parameter_selection.autocorrelation import autoCorrelation_tau
import numpy as np

fs = 10
t = np.linspace(0, 100, fs*100)
ts = np.sin(t) + np.sin((1/np.pi)*t)

tau = autoCorrelation_tau(ts, cutoff = 1/np.exp(1), AC_method = 'pearson', plotting = False)
print('Delay from AC: ', tau)


# In[ ]: PAMI
    
from teaspoon.parameter_selection.PAMI_delay import PAMI_for_delay 
import numpy as np

fs = 10
t = np.linspace(0, 100, fs*100)
ts = np.sin(t)
tau = PAMI_for_delay(ts, n = 3, plotting = True)
print('Delay from PAMI: ',tau)


# In[ ]: Fourier Spectrum Analysis
    
from teaspoon.parameter_selection.delay_LMS import LMSforDelay 
import numpy as np

fs = 10
t = np.linspace(0, 100, fs*100)
ts = np.sin(t) + np.random.normal(0,0.1, len(t))

tau = LMSforDelay(ts, fs, plotting = True)
print('Delay from Fourier: ' + str(int(tau)))


# In[ ]: MsPE for n and tau

import numpy as np
from teaspoon.parameter_selection.MsPE import MsPE_n,  MsPE_tau

t = np.linspace(0, 100, 1000)
ts = np.sin(t)

m_s, m_e, d_s, d_e = 3, 7, 1, 200
#m_s and m_e are the starting and ending dimensions n to search through
#d_e = max delay tau to search through

#plotting option will show you how delay tau or dimension n were selected
tau = int(MsPE_tau(ts, d_e, plotting = True))
n = MsPE_n(ts, tau, m_s, m_e, plotting = True)

print('Embedding Delay:       '+str(tau))
print('Embedding Dimension:   '+str(n))



# In[ ]: FNN

from teaspoon.parameter_selection.FNN_n import FNN_n
import numpy as np

fs = 10
t = np.linspace(0, 100, fs*100)
ts = np.sin(t)

tau=15 #embedding delay

perc_FNN, n = FNN_n(ts, tau, plotting = True)
print('FNN embedding Dimension: ',n)
