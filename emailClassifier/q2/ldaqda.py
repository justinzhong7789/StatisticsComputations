import numpy as np
import matplotlib.pyplot as plt
import util

# a helper matrix multiplication function just for 2x1 x 1x2
def matmul(x):
    result = np.zeros((2,2))
    result[0][0] = x[0]**2
    result[1][1] = x[1]**2
    result[0][1] = x[1]*x[0]
    result[1][0] = x[1]*x[0]
    return result

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """

    # Get mu's first by taking average
    d = len(x)
    allHeights = x[:, 0]
    allWeights = x[:, 1]
    totalHeight, totalWeight = (x.sum(axis=0)[i] for i in range(2))

    manInd, womanInd = [], []
    for i in range(d):
        if y[i] == 1: manInd.append(i)
        else: womanInd.append(i)
    
    manTotalHeight, manTotalWeight = (x[manInd, i].sum() for i in range(2))
    womanTotalHeight, womanTotalWeight = (x[womanInd, i].sum() for i in range(2))
    
    ## mu's calculations
    mu_male_height = manTotalHeight/len(manInd)
    mu_male_weight = manTotalWeight/len(manInd)
    mu_female_height = womanTotalHeight/len(womanInd)
    mu_female_weight = womanTotalWeight/len(womanInd)
    mu_height = totalHeight/d
    mu_weight = totalWeight/d

    mu_male = np.array([mu_male_height, mu_male_weight])
    mu_female = np.array([mu_female_height, mu_female_weight])
    mu = np.array([mu_height, mu_weight])

    # next get cov's
    #cov = np.zeros((2,2)
    cov_female, cov_male, cov = ((np.zeros((2,2))) for i in range(3))
    
    # compute the covariance matrices
    for i in range(d):
        cov += matmul((x[i]-mu))
        if y[i] == 1:
            cov_male += matmul((x[i]-mu_male))
        else:
            cov_female += matmul((x[i]-mu_female))
    cov = cov/d
    cov_female = cov_female/len(womanInd)
    cov_male = cov_male/len(manInd)
    
    xx, yy = np.meshgrid(np.arange(50, 80, 1), np.arange(80, 280 ,5))
    # get prob for both qda and lda
    LDA_male, LDA_female =  ((np.zeros(xx.shape)) for i in range(2))
    QDA_male, QDA_female =  ((np.zeros(xx.shape)) for i in range(2))
    for i in range(len(yy)):
        coords = np.array([xx[0], yy[i]]).transpose()
        prob = util.density_Gaussian(mu_male, cov, coords)
        LDA_male[i] = prob
        prob = util.density_Gaussian(mu_female, cov, coords)
        LDA_female[i] = prob
        prob = util.density_Gaussian(mu_male, cov_male, coords)
        QDA_male[i] = prob
        prob = util.density_Gaussian(mu_female, cov_female, coords)
        QDA_female[i] = prob        

    # plotting lda
    plt.xlim([50, 80])
    plt.ylim([80, 280])
    plt.plot(x[manInd, 0], x[manInd,1], 'b^', label='man')
    plt.plot(x[womanInd, 0], x[womanInd, 1], 'r^', label='woman')
    
    plt.legend(loc='upper right', numpoints=1)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.contour(xx, yy, LDA_male, colors='green')
    plt.contour(xx, yy, LDA_female, colors='orange')
    plt.contour(xx,yy, (LDA_male-LDA_female), levels=1, colors='black')
    plt.title("LDA")
    plt.savefig("lda.png")
    
    # clear image
    plt.clf()
    # plotting qda
    plt.xlim([50, 80])
    plt.ylim([80, 280])
    plt.plot(x[manInd, 0], x[manInd,1], 'b^', label='man')
    plt.plot(x[womanInd, 0], x[womanInd, 1], 'r^', label='woman')
    plt.legend(loc='upper right', numpoints=1)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title("QDA")
    plt.contour(xx, yy, QDA_male, colors='green')
    plt.contour(xx, yy, QDA_female, colors='orange')
    plt.contour(xx,yy, (QDA_male-QDA_female), levels=1, colors='black')
    plt.savefig("qda.png")

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    resultLDA, resultQDA = [], []
    numSamples = len(x)
    for data in x:
        data = np.expand_dims(data, axis=0)
        resultLDA.append(1 if util.density_Gaussian(mu_male, cov, data) >
                         util.density_Gaussian(mu_female, cov, data) else 2)
        resultQDA.append(1 if util.density_Gaussian(mu_male, cov_male, data)>
                         util.density_Gaussian(mu_female, cov_female, data) else 2)

    mis_lda = (resultLDA!=y).sum()/numSamples
    mis_qda = (resultQDA!=y).sum()/numSamples
    print("miss of lda is: {}".format(mis_lda))
    print("miss of qda is: {}".format(mis_qda))
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
