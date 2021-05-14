import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import math
import numpy as np

# global variables
vocab = {}
num = '<num>'

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """

    
    # p_d = p(w_d | y_n=1) spam
    # q_d = p(w_d | y_n=0) ham

    vocab[num] = 0
    
    for file in file_lists_by_category[1]:
        for word in util.get_words_in_file(file):
            # this adds the new word to the vocab dict
            if word.isnumeric():
                vocab[num] = 0
            else:  vocab[word] = 0

    for file in file_lists_by_category[0]:
        for word in util.get_words_in_file(file):
            # this adds the new word to the vocab dict
            if word.isnumeric():
                vocab[num] = 0
            else:  vocab[word] = 0
    
    # add an entry to vocab dict to account for any new words from test set
    vocab["<unk>"] = 0
    
    # vocab contains all the words from the training set
    # but all the values are 0
    D = len(vocab)

    # HAM distribution 
    hamWords = vocab.copy()
    hamTotalWordCount = 0
    for file in file_lists_by_category[1]:
        for word in util.get_words_in_file(file):
            if word.isnumeric(): hamWords[num]+=1
            else: hamWords[word] += 1 
            hamTotalWordCount+=1
    
    # smoothe each element
    for element in hamWords:
        old_val = hamWords[element]
        hamWords[element] = (old_val+1)/(hamTotalWordCount+D)
    
    # SPAM distribution #
    spamWords = vocab.copy()
    spamTotalWordCount = 0
    for file in file_lists_by_category[0]:    
        for word in util.get_words_in_file(file):
            if word.isnumeric(): spamWords[num]+=1
            else: spamWords[word] += 1 
            spamTotalWordCount+=1

    # smoothe each elemetn
    for element in spamWords:
        old_val = spamWords[element]
        spamWords[element] = (old_val+1)/(spamTotalWordCount+D)
        
    return spamWords, hamWords   


def classify_new_email(filename,probabilities_by_category,prior_by_category, decisionFactor=1):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution
    Optional argument decisionFactor is 1 by default. decisionFactor > 1 will let
    the model make decision in favour of HAM. 

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    # indexing is confusing, so I give them variable name to be more intuitive
    spamDistribution, hamDistribution = probabilities_by_category

    # build the function feature vector from the mail first
    mailFeatureVec = vocab.copy()
    for word in util.get_words_in_file(filename):
        # word is a regular word
        if word in mailFeatureVec:
            mailFeatureVec[word] += 1
        # word represents a numeric value
        elif word.isnumeric(): mailFeatureVec[num] += 1
        # word is not recognized
        else:  mailFeatureVec["<unk>"] += 1
    
    
    spamProb = 0
    hamProb = 0
    for word in mailFeatureVec:
        # compute P(y=spam| mailFeatureVec)
        spamProb += mailFeatureVec[word]  * math.log(spamDistribution[word])
        # compute P(y=ham| mailFeatureVec)
        hamProb += mailFeatureVec[word] * math.log(hamDistribution[word])
    
    # multiply by prior distribution
    spamProb += math.log(prior_by_category[0])
    hamProb += math.log(prior_by_category[1])
    hamProb *= decisionFactor
    result = "ham" if (hamProb>spamProb) else "spam"
    return (result, (spamProb, hamProb)) #classify_result




if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    
    # generate a list of decision factors, from 0.6 to 1.5
    decisionFactorList = np.arange(0.8, 1.3, 0.05)

    # make a list of tuples to contain type 1 and type 2 errors
    numType1 = []
    numType2 = []

    for i in range(len(decisionFactorList)):
        performance = np.zeros([2, 2])
        for filename in (util.get_files_in_folder(test_folder)):
            label, log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category,
                                                 decisionFactor=decisionFactorList[i])
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance[int(true_index), int(guessed_index)] += 1
        correct = np.diag(performance)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance, 1)
        numType1.append(totals[0] - correct[0])
        numType2.append(totals[1] - correct[1])
        print("Iteration: {}. Decision factor is: {}".format(i+1,decisionFactorList[i]))
        print(template % (correct[0],totals[0],correct[1],totals[1]))

    print("number of type 1 error list is:")
    print(numType1)
    print("number of type 2 error list is:")
    print(numType2)

    plt.plot(numType1, numType2, 'ro')
    plt.ylabel("number of type2 error")
    plt.xlabel("number of type1 error")
    plt.title("type 2 vs type 1 error")
    plt.savefig("figure.png")

