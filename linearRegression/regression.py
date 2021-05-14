import numpy as np
import matplotlib.pyplot as plt
import util


    

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    mu = np.array([0, 0]).reshape((2,1))
    cov = np.eye(2)*beta
    xx, yy = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1 ,0.01))
    x_set = np.dstack((xx, yy))
    x_set = np.reshape(x_set, (len(xx)*len(yy), 2))
    prob = util.density_Gaussian(mu.transpose(), cov, x_set).reshape(xx.shape[0], yy.shape[1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.plot(-0.1, -0.5, "bo")
    plt.contour(xx, yy, prob)
    plt.title("prior")
    plt.savefig("prior.png")
    return 
    
def posteriorDistribution(x,z,beta,sigma2, imgname):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    inv = np.linalg.inv
    det = np.linalg.det
    N = len(x)

    A = np.vstack((np.zeros((1, N))+1, x.transpose()))
    A = A.transpose()
    sigma_a = np.eye(2)*(1/beta)
    sigma_w = 1/sigma2

    
    mid = inv(sigma_a + np.matmul(A.transpose()*sigma_w, A))
    mu = np.matmul(inv(sigma_a + np.matmul(A.transpose()* sigma_w, A)), 
                    np.matmul(A.transpose()*sigma_w, z))

    cov = inv(sigma_a+ np.matmul(A.transpose()*sigma_w, A))
    
    
    xx, yy = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1 ,0.01))
    x_set = np.dstack((xx, yy))
    x_set = np.reshape(x_set, (len(xx)*len(yy), 2))
    prob = util.density_Gaussian(mu.transpose(), cov, x_set).reshape(yy.shape[0], xx.shape[1])
    
    plt.clf()
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.title("posterior" + str(N))
    plt.plot(-0.1, -0.5,'bo')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.contour(xx, yy, prob)
    plt.savefig(imgname)

    return (mu,cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train, numPredictions):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    sigma_a = np.eye(2)*(1/beta)
    N = len(x)
    x_np = np.expand_dims(np.array(x) , 0)
    A = np.vstack(((np.zeros((1, N))+1), x_np)).transpose()
    mu_z = np.matmul(A, mu)
    sigma_z = sigma2 + np.matmul(np.matmul(A, Cov), A.transpose())
    std_z = np.sqrt(np.diag(sigma_z))
    

    plt.clf()
    plt.title("predict" + str(numPredictions))
    plt.scatter(x_train, z_train, color = 'blue')
    
    
    plt.errorbar(x, mu_z, yerr=std_z, fmt='ro')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig("predict"+str(numPredictions)+".png")
    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    

    # prior distribution p(a)
    priorDistribution(beta)
    

    mu, cov = [], []
    numPredictions = [1, 5, 100]
    for i in numPredictions:
        x = x_train[0:i]
        z = z_train[0:i]
        # posterior distribution p(a|x,z)
        imgname = "posterior" + str(i) + ".png"
        m, c = posteriorDistribution(x,z,beta,sigma2, imgname)
        mu.append(m)
        cov.append(c)

    # distribution of the prediction
    for i in range(3):
        #imgname = "predict"+ str(numPredictions[i]) + ".png"
        predictionDistribution(x_test, beta, sigma2, mu[i], cov[i],
                                x_train[ :numPredictions[i]],
                                z_train[ :numPredictions[i]], numPredictions[i])


   

    
    
    

    
