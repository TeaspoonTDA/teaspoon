"""
False Nearest Neighbors (FNN) for dimension (n).
==================================================


"""

def ts_recon(ts, dim, tau):
    import numpy as np
    xlen = len(ts)-(dim-1)*tau
    a= np.linspace(0,xlen-1,xlen)
    a= np.reshape(a,(xlen,1))
    delayVec=np.linspace(0,(dim-1),dim)*tau
    delayVec= np.reshape(delayVec,(1,dim))
    delayMat=np.tile(delayVec,(xlen,1))
    vec=np.tile(a,(1,dim))
    indRecon = np.reshape(vec,(xlen,dim)) + delayMat
    indRecon = indRecon.astype(np.int64)
    tsrecon = ts[indRecon]
    tsrecon = tsrecon[:,:,0]
    return tsrecon


def FNN_n(ts, tau, maxDim = 10, plotting = False, Rtol=15, Atol=2, threshold = 10):
    """This function implements the False Nearest Neighbors (FNN) algorithm described by Kennel et al. 
    to select the minimum embedding dimension.

    Args:
       ts (array):  Time series (1d).
       tau (int):  Embedding delay.
       
       
    Kwargs:
       maxDim (int):  maximum dimension in dimension search. Default is 10.
       
       plotting (bool): Plotting for user interpretation. Defaut is False.
       
       Rtol (float): Ratio tolerance. Defaut is 15.
       
       Atol (float): A tolerance. Defaut is 2.
       
       threshold (float): Tolerance threshold for percent of nearest neighbors. Defaut is 10.
       
    Returns:
       (int): n, The embedding dimension.
       
    """

    import numpy as np
    from scipy.spatial import KDTree
    if len(ts)-(maxDim-1)*tau < 20:
        maxDim=len(ts)-(maxDim-1)*tau-1
    ts = np.reshape(ts, (len(ts),1)) #ts is a column vector
    st_dev=np.std(ts) #standart deviation of the time series
    
    Xfnn=[]
    dim_array = []
    
    flag = False
    i = 0
    while  flag == False:
        i = i+1
        dim=i
        tsrecon = ts_recon(ts, dim, tau)#delay reconstruction
        
        tree=KDTree(tsrecon)
        D,IDX=tree.query(tsrecon,k=2)
        
        #Calculate the false nearest neighbor ratio for each dimension
        if i>1:
            D_mp1=np.sqrt(np.sum((np.square(tsrecon[ind_m,:]-tsrecon[ind,:])),axis=1))
            #Criteria 1 : increase in distance between neighbors is large
            num1 = np.heaviside(np.divide(abs(tsrecon[ind_m,-1]-tsrecon[ind,-1]),Dm)-Rtol,0.5)
            #Criteria 2 : nearest neighbor not necessarily close to y(n)
            num2= np.heaviside(Atol-D_mp1/st_dev,0.5)
            num=sum(np.multiply(num1,num2))
            den=sum(num2)
            Xfnn.append((num/den)*100)
            dim_array.append(dim-1)
            if (num/den)*100 <= 10 or i == maxDim:
                flag = True
                
        # Save the index to D and k(n) in dimension m for comparison with the
        # same distance in m+1 dimension   
        xlen2=len(ts)-dim*tau
        Dm=D[0:xlen2,-1]
        ind_m=IDX[0:xlen2,-1]
        ind=ind_m<=xlen2-1
        ind_m=ind_m[ind]
        Dm=Dm[ind]
    Xfnn = np.array(Xfnn)
        
    if plotting == True:
        import matplotlib.pyplot as plt
        TextSize = 14
        plt.figure(1) 
        plt.plot(dim_array, Xfnn)
        plt.xlabel(r'Dimension $n$', size = TextSize)
        plt.ylabel('Percent FNN', size = TextSize)
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.ylim(0)
        plt.savefig('C:\\Users\\myersau3.EGR\\Desktop\\python_png\\FNN_fig.png', bbox_inches='tight',dpi = 400)
        plt.show()
    
    return Xfnn, dim-1



# In[ ]:


if __name__ == '__main__':
    
    import numpy as np
    
    fs = 10
    t = np.linspace(0, 100, fs*100) 
    ts = np.sin(t)

    tau=15 #embedding delay

    perc_FNN, n =FNN_n(ts, tau, plotting = True)
    print('FNN embedding Dimension: ',n)
    
    
          
        
        