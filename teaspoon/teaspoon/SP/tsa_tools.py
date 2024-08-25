from sklearn.neighbors import NearestNeighbors
from teaspoon.parameter_selection import MsPE
import matplotlib.pyplot as plt
from teaspoon.parameter_selection import FNN_n
from teaspoon.parameter_selection import MI_delay
import numpy as np
import itertools
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# In[ ]:

def cgss_binning(ts, n=None, tau=None, b=12, binning_method='equal_size', plot_binning=False):
    """This function generates the binning array applied to each dimension when constructing CGSSN.

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay from state space reconstruction. Default uses FNN algorithm from parameter_selection module.
        b (Optional[int]): Number of bins per dimension. Default is 12.
        plot_binning (Optional[bool]): Option to display plotting of binning over space occupied by SSR. default is False.
        binning_method (Optional[str]): Either equal_size or equal_probability for creating binning array. Default is equal_size.

    Returns:
        [1-D array]: One-dimensional array of bin edges.
    """
    B = b
    # ----------------Get embedding parameters if not defined----------------
    if tau == None:
        tau = MI_delay.MI_for_delay(
            ts, method='basic', h_method='sturge', k=2, ranking=True)
    if n == None:
        perc_FNN, n = FNN_n.FNN_n(ts, tau)

    def equalObs(x, B):  # define function to calculate equal-frequency bins based on interpolation
        return np.interp(np.linspace(0, len(x), B), np.arange(len(x)), np.sort(x))

    # get state space reconstruction from signal (SSR)
    from teaspoon.SP.tsa_tools import takens
    SSR = takens(ts, n, tau)

    # ----------------Define how to use the embedding----------------
    embedding = np.array(SSR).T

    # ----------------Define how to partition the embedding----------------
    if binning_method == 'equal_frequency':
        # define bins with equal-frequency or probability (approximately)
        B_array = equalObs(embedding.flatten(), B+1)
        B_array[-1] = B_array[-1]*(1 + 10**-10)
    if binning_method == 'equal_size':  # define bins based on equal spacing
        B_array = np.linspace(np.amin(embedding),
                              np.amax(embedding)*(1+10**-10), B+1)

    if plot_binning == True:
        x_max, x_min = np.max(B_array), np.min(B_array)
        plt.figure(0)
        plt.figure(figsize=(6, 6))

        TextSize = 17
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.xlabel('$x(t)$', size=TextSize)
        plt.ylabel(r'$x(t + \tau)$', size=TextSize)
        plt.scatter(SSR.T[0], SSR.T[1], c='k', marker='o',
                    alpha=0.2, label='$v_i$', s=10.)
        for b in B_array:
            plt.plot([x_max, x_min], [b, b], 'k--', linewidth=1)
            plt.plot([b, b], [x_max, x_min], 'k--', linewidth=1)
        plt.show()

    return B_array


def cgss_sequence(SSR, B_array):
    """This function generates the sequence of coarse-grained state space states from the state space reconstruction.

    Args:
        SSR (n-D array): n-dimensional state state space reconstruction using takens function from teaspoon.
        B_array (1-D array): binning array from function cgss_binning or self defined. Array of bin edges applied to each dimension of SSR for CGSS state sequence.

    Returns:
        [1-D array of integers]: array of coarse grained state space states represented as ints.
    """

    embedding = SSR.T

    B = len(B_array) - 1
    n = len(SSR[0])
    basis = B**(np.arange(n))  # basis for assigning symbolic value

    # ----------------digitize the embedding to a sequence----------------
    digitized_embedding = []  # prime the digitized version of deltas
    for e_i in embedding:  # nloop through n-1 delta positions
        # digitalize column delta_i
        digitzed_vector = np.digitize(e_i, bins=B_array)
        # append to digitalized deltas data structure
        digitized_embedding.append(digitzed_vector)
    # digitalize and stacked delta vectors
    digitized_embedding = np.array(digitized_embedding).T - 1
    # symbolic sequence from basis and digitzed embedding
    symbol_seq = np.sum(np.array(basis)*digitized_embedding, axis=1)

    return symbol_seq


# finds permutation sequency from modified pyentropy package
def permutation_sequence(ts, n=None, tau=None):
    """This function generates the sequence of permutations from a 1-D time series.

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MsPE algorithm from parameter_selection module.

    Returns:
        [1-D array of intsegers]: array of permutations represented as int from [0, n!-1] from the time series.
    """

    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    time_series = ts

    if n == None:
        from teaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts))
        n = MsPE.MsPE_n(ts, tau)

    if tau == None:
        from teaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts))

    m, delay = n, tau

    import itertools
    import numpy as np

    def util_hash_term(perm):  # finds permutation type
        deg = len(perm)
        return sum([perm[k]*deg**k for k in range(deg)])
    L = len(time_series)  # total length of time series
    perm_order = []  # prepares permutation sequence array
    # prepares all possible permutations for comparison
    permutations = np.array(list(itertools.permutations(range(m))))
    hashlist = [util_hash_term(perm)
                for perm in permutations]  # prepares hashlist
    for i in range(L - delay * (m - 1)):
        # For all possible permutations in time series
        sorted_index_array = np.array(np.argsort(
            time_series[i:i + delay * m:delay], kind='quicksort'))
        # sort array for catagorization
        hashvalue = util_hash_term(sorted_index_array)
        # permutation type
        perm_order = np.append(
            perm_order, np.argwhere(hashlist == hashvalue)[0][0])
        # appends new permutation to end of array
    # sets permutation type as integer where $p_i \in \mathbb{z}_{>0}$
    perm_seq = perm_order.astype(int)+1
    return perm_seq  # returns sequence of permutations


# In[ ]:


def takens(ts, n=None, tau=None):
    """This function generates an array of n-dimensional arrays from a time-delay state-space reconstruction.

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses FNN algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MI algorithm from parameter_selection module.

    Returns:
        [arraay of n-dimensional arrays]: array of delyed embedded vectors of dimension n for state space reconstruction.
    """

    import numpy as np
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    if tau == None:
        from teaspoon.parameter_selection import MI_delay
        tau = MI_delay.MI_for_delay(
            ts, method='basic', h_method='sturge', k=2, ranking=True)
    if n == None:
        from teaspoon.parameter_selection import FNN_n
        perc_FNN, n = FNN_n.FNN_n(ts, tau)

    # takens embedding method. Not the fastest algoriothm, but it works. Need to improve
    L = len(ts)  # total length of time series
    SSR = []
    for i in range(n):
        SSR.append(ts[i*tau:L-(n-i-1)*tau])
    SSR = np.array(SSR).T

    return np.array(SSR)


# In[ ]:


def k_NN(embedded_time_series, k=4):
    """This function gets the k nearest neighbors from an array of the state space reconstruction vectors

    Args:
        embedded_time_series (array of n-dimensional arrays): state space reconstructions vectors of dimension n. Can use takens function.

    Other Parameters:
        k (Optional[int]): number of nearest neighbors for graph formation. Default is 4.

    Returns:
        [distances, indices]: distances and indices of the k nearest neighbors for each vector.
    """

    ETS = embedded_time_series
    from sklearn.neighbors import NearestNeighbors
    # get nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(ETS)
    # get incidices of nearest neighbors
    distances, indices = nbrs.kneighbors(ETS)
    return distances, indices


# In[ ]:

def ZeDA(sig, t1, tn, level=0.0,  plotting=False, method='std', score=3.0):
    """This function takes a uniformly sampled time series and finds level-crossings.

    Args:
        sig (numpy array):   Time series (1d) in format of npy.
        t1 (float):          Initial time of recording signal.
        tn (float):          Final time of recording signal.

    Other Parameters:
        level (Optional[float]):  Level at which crossings are to be found; default: 0.0 for zero-crossings
        plotting (Optional[bool]): Plots the function with returned brackets; defaut is False.
        method (Optional[str]): Method to use for setting persistence threshold; 'std' for standard deviation, 'iso' for isolation forest, 'dt' for smallest time interval in case of a clean signal; default is std (3*standard deviation)
        score (Optional[float]): z-score to use if method == 'std'; default is 3.0

    Returns:
        [tuples, list, list]: brackets gives intervals in the form of tuples where a crossing is expected; crossings gives the estimated crossings in each interval obtained by averaging both ends; flags marks, for each interval, whether both ends belong to the same sgn(function) category (0, unflagged; 1, flagged)

    """

    ##############################################################
    # Sorts time series in P and Q point clouds
    def sortPQ(sig, t):

        p = np.sort(t[np.sign(sig) == 1]).tolist()
        q = np.sort(t[np.sign(sig) == -1]).tolist()

        # Adding endpoints if not already there
        if t[0] not in p:
            p.append(t[0])
        if t[0] not in q:
            q.append(t[0])
        if t[-1] not in p:
            p.append(t[-1])
        if t[-1] not in q:
            q.append(t[-1])

        p.sort()
        q.sort()

        return p, q

    ################################################################
    # Find points in persistence diagram above the threshold mu
    def sortPQ_mu(p, q, zscore=score):

        import statistics

        p_mu = []
        q_mu = []

        if method == 'std':
            p_per = np.diff(p)
            q_per = np.diff(q)
            lst = list(p_per) + list(q_per)
            dev = statistics.stdev(lst)
            mu = zscore * dev
        elif method == 'dt':
            mu = 1.9*(tn - t1)/np.size(sig)

        n = 0
        while n < len(p) - 1:
            if p[n + 1] - p[n] > mu:
                p_mu.append(p[n])
                p_mu.append(p[n + 1])
            n = n + 1

        n = 0
        while n < len(q) - 1:
            if q[n + 1] - q[n] > mu:
                q_mu.append(q[n])
                q_mu.append(q[n + 1])
            n = n + 1

        p_mu = list(set(p_mu))
        p_mu.sort()
        q_mu = list(set(q_mu))
        q_mu.sort()

        return p_mu, q_mu

    ################################################################
    # Find outliers in persistence diagram using Isolation Forest
    def sortPQ_outlier(p, q):

        p_mu = []
        q_mu = []

        p_per = np.diff(p)
        q_per = np.diff(q)

        y = list(p_per) + list(q_per)
        X = list(range(0, len(y)))

        import pandas as pd
        from sklearn.ensemble import IsolationForest

        data_values = np.array(list(zip(X, y)))

        data = pd.DataFrame(data_values, columns=['x', 'y'])

        def fit_model(model, data, column='y'):
            df = data.copy()
            data_to_predict = data[column].to_numpy().reshape(-1, 1)
            predictions = model.fit_predict(data_to_predict)
            df['Predictions'] = predictions
            return df

        iso_forest = IsolationForest(n_estimators=125)
        iso_df = fit_model(iso_forest, data)
        iso_df['Predictions'] = iso_df['Predictions'].map(
            lambda x: 1 if x == -1 else 0)

        predictions = iso_df['Predictions'].to_numpy()

        cat_p = predictions[0:len(p_per) - 1]
        cat_q = predictions[len(p_per):len(p_per) + len(q_per) - 1]

        for i in range(0, len(predictions)):
            if i < len(p_per):
                if predictions[i] == 1:
                    p_mu.append(p[i])
                    p_mu.append(p[i+1])

            if i >= len(p_per):
                if predictions[i] == 1:
                    q_mu.append(q[i - len(p_per)])
                    q_mu.append(q[i - len(p_per) + 1])

        return p_mu, q_mu

    ####################################################################
    # Interleave P_mu and Q_mu
    def interleave(p_mu, q_mu):

        import operator

        I_index = [*range(0, len(p_mu))] + \
            [*range(len(p_mu), len(p_mu) + len(q_mu))]
        I = p_mu + q_mu

        I_dict = dict(zip(I_index, I))

        if len(p_mu) == 0:
            if t[0] != I_dict[0]:
                I_dict[-1] = t[0]
            else:
                del I_dict[0]
            if t[len(t) - 1] != I_dict[len(q_mu) - 1]:
                I_dict[len(q_mu)] = t[len(t) - 1]
            else:
                del I_dict[len(q_mu) - 1]

        if len(q_mu) == 0:
            if t[0] != I_dict[0]:
                I_dict[-1] = t[0]
            else:
                del I_dict[0]
            if t[len(t) - 1] != I_dict[len(p_mu) - 1]:
                I_dict[len(p_mu)] = t[len(t) - 1]
            else:
                del I_dict[len(p_mu) - 1]

        if len(p_mu) != 0 and len(q_mu) != 0:
            if t[0] != I_dict[0] and t[0] != I_dict[len(p_mu)]:
                I_dict[-1] = t[0]
            else:
                if t[0] == I_dict[0] and t[0] == I_dict[len(p_mu)]:
                    del I_dict[0]
                    del I_dict[len(p_mu)]
                elif t[0] == I_dict[0]:
                    del I_dict[0]
                else:
                    del I_dict[len(p_mu)]

            if t[len(t) - 1] != I_dict[len(p_mu) - 1] and t[len(t) - 1] != I_dict[len(p_mu) + len(q_mu) - 1]:
                I_dict[len(p_mu) + len(q_mu)] = t[len(t) - 1]
            else:
                if t[len(t) - 1] == I_dict[len(p_mu) - 1] and t[len(t) - 1] == I_dict[len(p_mu) + len(q_mu) - 1]:
                    del I_dict[len(p_mu) - 1]
                    del I_dict[len(p_mu) + len(q_mu) - 1]
                elif t[len(t) - 1] == I_dict[len(p_mu) - 1]:
                    del I_dict[len(p_mu) - 1]
                else:
                    del I_dict[len(p_mu) + len(q_mu) - 1]

        sorted_tuples = sorted(I_dict.items(), key=operator.itemgetter(1))
        I_dict = {k: v for k, v in sorted_tuples}

        brackets = list(zip(*[iter(I_dict.values())] * 2))

        n = 0
        ZC = []
        flag = []
        for n in range(0, len(brackets)):

            k1 = [i for i in I_dict if I_dict[i] == brackets[n][0]]
            k2 = [i for i in I_dict if I_dict[i] == brackets[n][1]]
            t1 = k1[0] in range(0, len(p_mu))
            t2 = k2[0] in range(len(p_mu), len(p_mu) + len(q_mu))

            root = 0.5 * (brackets[n][0] + brackets[n][1])
            ZC.append(root)

            if (t1 ^ t2):
                flag.append(1)
            else:
                flag.append(0)

        return brackets, ZC, flag

    #########################################################################
    # ZeDA Function

    import numpy as np

    sig = sig - level
    t = np.linspace(t1, tn, np.size(sig), endpoint=True)

    p, q = sortPQ(sig, t)

    if len(p) == len(t) or len(q) == len(t):
        print("The signal does not appear to have any crossings.")
        exit()

    if method == 'std':
        p_mu, q_mu = sortPQ_mu(p, q)
        c = 0
        while len(p_mu) == 0 and len(q_mu) == 0:
            print("The z-score is high, lowering it by 0.25.")
            score = score - 0.25
            p_mu, q_mu = sortPQ_mu(p, q, score)
            c = c+1
        if c > 0:
            print(f"z-score used is to find outliers is {score}.")

    elif method == 'dt':
        p_mu, q_mu = sortPQ_mu(p, q)

    elif method == 'iso':
        p_mu, q_mu = sortPQ_outlier(p, q)
        if len(p_mu) == 0 and len(q_mu) == 0:
            print(
                "The method 'iso' could not detect any points. Please use another method.")

    else:
        print('Keyword for method unrecognized. Use "std" for standard deviation, "iso" for Isolation Forest, or "dt" for signal with no noise.')
        exit()

    brackets, ZC, Flag = interleave(p_mu, q_mu)

    #######################################################################
    # Plot Everything, if plotting == True
    if plotting == True:

        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.figure(figsize=(8, 4), dpi=200)

        plt.plot(t, sig, linewidth=1.5, alpha=1)
        plt.axhline(y=0, linewidth=0.75, linestyle=':',
                    color='grey', label='_nolegend_')
        plt.xlim([t1, tn])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Time", fontsize=25)
        plt.ylabel("Amplitude", fontsize=25)

        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(brackets, ymin=ymin, ymax=ymax,
                   linewidth=1.0, color='green', linestyle='--')

        n = np.zeros(len(ZC))
        plt.scatter(ZC, n, s=150, color='black', marker='x', zorder=2)
        plt.legend(['Signal', 'Brackets', 'Roots'])

        plt.tight_layout()
        plt.show()

    ##################################################################
    # Return Brackets, Crossings and Flags
    print("Brackets are: ", brackets)
    print("Estimated crossings are: ", ZC)
    print("Corresponding flags are: ", Flag)

    return brackets, ZC, Flag

##########################################################################

# In[ ]:


def first_zero(sig, t1, tn, level=0.0, r=1.2, plotting=False):
    """This function takes a discrete time series and finds the first zero-crossing or the global minimum (if no crossing).

    Args:
        sig (numpy array):   Time series (1d) in format of npy.
        t1 (float):          Initial time of recording signal.
        tn (float):          Final time of recording signal.

    Other Parameters:
        level (Optional[float]):  Level at which crossings are to be found; default: 0.0 for zero-crossings
        r (Optional[float]):  Reliability (>0)--increasing r increases reliability of the method but may result in a higher convergence time; default: 1.2
        plotting (Optional[bool]): Plots the function with returned brackets; defaut is False.

    Returns:
        float: the estimated first crossing in the interval or the global minimum.

    """

    if r < 0:
        print('Reliability must be > 0.')
        return

    epsilon = (tn - t1)/(len(sig)-1)  # Convergence Criterion

    ##############################################################
    # Finding t after t^(a) and t^(b)

    x = [t[0], t[len(t) - 1]]
    y = [sig[0], sig[len(sig) - 1]]
    s1 = abs(y[1] - y[0])/(x[1] - x[0])
    lambda1 = s1
    lambda_max2 = s1
    del_t_max2 = x[1] - x[0]
    lambda2 = lambda_max2*(x[1] - x[0]) / del_t_max2
    m1 = r*max(lambda1, lambda2, epsilon)
    R1 = 0.5*(y[0]+y[1]) - 0.5*m1*(x[1] - x[0])
    t_hat1 = 0.5*(x[1] + x[0] - (y[1] - y[0])/m1)

    if R1 <= 0:
        t_star = t[0] + sig[0]/m1
        t_new = t[np.argmin(abs(t - t_star))]
        if sig[np.argmin(abs(t - t_star))] <= 0:
            x.remove(t[len(t) - 1])
            y.remove(sig[len(sig) - 1])
            x.append(t_new)
            y.append(sig[np.argmin(abs(t - t_star))])
            zipped_pairs = zip(x, y)
            sorted_pairs = sorted(zipped_pairs)
            x = [item[0] for item in sorted_pairs]
            y = [item[1] for item in sorted_pairs]
        else:
            x.append(t_new)
            y.append(sig[np.argmin(abs(t - t_star))])
            zipped_pairs = zip(x, y)
            sorted_pairs = sorted(zipped_pairs)
            x = [item[0] for item in sorted_pairs]
            y = [item[1] for item in sorted_pairs]
    else:
        t_new = t[np.argmin(abs(t - t_hat1))]
        if sig[np.argmin(abs(t - t_hat1))] <= 0:
            x.remove(t[len(t) - 1])
            y.remove(sig[len(sig) - 1])
            x.append(t_new)
            y.append(sig[np.argmin(abs(t - t_hat1))])
            zipped_pairs = zip(x, y)
            sorted_pairs = sorted(zipped_pairs)
            x = [item[0] for item in sorted_pairs]
            y = [item[1] for item in sorted_pairs]
        else:
            x.append(t_new)
            y.append(sig[np.argmin(abs(t - t_hat1))])
            zipped_pairs = zip(x, y)
            sorted_pairs = sorted(zipped_pairs)
            x = [item[0] for item in sorted_pairs]
            y = [item[1] for item in sorted_pairs]

    ######################################################################
    # nth iteration

    n = 3
    dt = abs(t1 - t_new)

    while dt > epsilon:

        x1 = []
        y1 = []
        s = []
        lambda1 = []
        lambda2 = []
        lambda_max = 0
        del_t_max = 0
        m = []
        R = []
        t_hat = []
        t_new_past = t_new
        t_new = 0

        # Find intervals
        for i in range(0, len(x)-1):
            x1.append([x[i], x[i+1]])
            y1.append([y[i], y[i+1]])

        # Find S for each interval
        for i in range(0, len(x1)):
            s.append(abs(y1[i][1] - y1[i][0])/(x1[i][1] - x1[i][0]))

        # Find lambda' for each interval
        for i in range(0, len(s)):
            if i == 0 and len(s) < 2:
                lambda1.append(s[i])
            elif i == 0:
                lambda1.append(max(s[i], s[i+1]))
            elif i == 1 and len(s) == 1:
                lambda1.append(max(s[i-1], s[i]))
            elif i == len(s) - 1 and len(s) > 0:
                lambda1.append(max(s[i-1], s[i]))
            else:
                lambda1.append(max(s[i-1], s[i], s[i+1]))

        # Find lambda_max and delta_t max for this trial
        lambda_max = max(s[:])
        if len(x1) < 2:
            del_t_max = x1[0][1] - x1[0][0]
        else:
            t_2 = np.array(x1[:][1])
            t_1 = np.array(x1[:][0])
            del_t_max = max(np.subtract(t_2, t_1))

        # Find lambda'', m, R and t_hat for each interval
        for i in range(0, len(x1)):
            lambda2.append(lambda_max*(x1[i][1] - x1[i][0]) / del_t_max)
            m.append(r*max(lambda1[i], lambda2[i], epsilon))
            R.append(0.5*(y1[i][1] + y1[i][0] - m[i]*(x1[i][1] - x1[i][0])))
            t_hat.append(0.5*(x1[i][1] + x1[i][0] -
                         (y1[i][1] - y1[i][0])/m[i]))

        cond = [i for i in range(len(R)) if R[i] <= 0]

        if cond:
            t_star = x1[cond[0]][0] + y1[cond[0]][0] / m[cond[0]]
            t_TauMinOne = x1[cond[0]][0]
            t_new = t[np.argmin(abs(t - t_star))]
        else:
            i = np.argmin(np.array(R))
            t_TauMinOne = x1[i][0]
            t_new = t[np.argmin(abs(t - t_hat[i]))]

        if sig[np.where(t == t_new)[0][0]] <= 0:
            x.remove(x[len(x) - 1])
            y.remove(y[len(y) - 1])
            if t_new not in x:
                x.append(t_new)
                y.append(sig[np.where(t == t_new)[0][0]])
        else:
            if t_new not in x:
                x.append(t_new)
                y.append(sig[np.where(t == t_new)[0][0]])

        zipped_pairs = zip(x, y)
        sorted_pairs = sorted(zipped_pairs)
        x = [item[0] for item in sorted_pairs]
        y = [item[1] for item in sorted_pairs]

        n = n + 1
        dt = abs(t_new - t_TauMinOne)

        if t_new == t_new_past:
            break

    print(f"Final root is {t_new} with a convergence criterion of {epsilon}.")

    if plotting == True:

        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.figure(figsize=(8, 4), dpi=200)

        plt.plot(t, sig, linewidth=1.5, alpha=1)
        plt.axhline(y=0, linewidth=0.75, linestyle=':',
                    color='grey', label='_nolegend_')
        plt.xlim([t1, tn])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Time", fontsize=25)
        plt.ylabel("Amplitude", fontsize=25)

        plt.scatter(t_new, 0, s=150, color='black', marker='x', zorder=2)
        plt.legend(['Signal', 'First Root'])

        plt.tight_layout()
        plt.show()

    return t_new


# In[ ]:

# Only runs if running from this file (This will show basic example)
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # ------------------------------------TAKENS-----------------------------------------

    import numpy as np
    t = np.linspace(0, 30, 200)
    ts = np.sin(t)  # generate a simple time series

    from teaspoon.SP.tsa_tools import takens
    embedded_ts = takens(ts, n=2, tau=10)

    import matplotlib.pyplot as plt
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    plt. show()

    # ------------------------------------PS-----------------------------------------

    import numpy as np
    t = np.linspace(0, 30, 200)
    ts = np.sin(t)  # generate a simple time series

    from teaspoon.SP.tsa_tools import permutation_sequence
    PS = permutation_sequence(ts, n=3, tau=12)

    import matplotlib.pyplot as plt
    plt.plot(t[:len(PS)], PS, 'k')
    plt. show()

    # ------------------------------------kNN-----------------------------------------

    import numpy as np
    t = np.linspace(0, 15, 100)
    ts = np.sin(t)  # generate a simple time series

    from teaspoon.SP.tsa_tools import takens
    embedded_ts = takens(ts, n=2, tau=10)

    from teaspoon.SP.tsa_tools import k_NN
    distances, indices = k_NN(embedded_ts, k=4)

    import matplotlib.pyplot as plt
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    i = 20  # choose arbitrary index to get NN of.
    NN = indices[i][1:]  # get nearest neighbors of point with that index.
    plt.plot(embedded_ts.T[0][NN], embedded_ts.T[1][NN], 'rs')  # plot NN
    plt.plot(embedded_ts.T[0][i], embedded_ts.T[1]
             [i], 'bd')  # plot point of interest
    plt. show()
