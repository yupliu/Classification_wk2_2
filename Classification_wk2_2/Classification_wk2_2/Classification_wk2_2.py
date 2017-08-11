from __future__ import division
import graphlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

try:
    import seaborn
except ImportError:
    pass

products = graphlab.SFrame('C:\\Machine_Learning\\Classification_wk2\\amazon_baby_subset.gl\\')
print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

import json
with open('C:\\Machine_Learning\\Classification_wk2\\important_words.json','r') as f:
    important_words = json.load(f)
important_words = [str(s) for s in important_words]
print important_words

def remove_punctuation(text):
    return text.translate(None,string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s:s.split().count(word))
products['perfect']
products['contains_perfect'] = products['perfect'].apply(lambda x: 1 if x >= 1 else 0)
print products['contains_perfect'].sum()

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

# Warning: This may take a few minutes...
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment') 
feature_matrix.shape
len(important_words)

'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # YOUR CODE HERE
    wh = np.dot(feature_matrix,coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    predictions = 1 / ( 1 + np.exp(-wh))
    #print predictions
    # return predictions
    return predictions

dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_predictions           =', correct_predictions
print 'output of predict_probability =', predict_probability(dummy_feature_matrix,dummy_coefficients)

def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)    
    # Return the derivative
    return derivative

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp

dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])

correct_indicators  = np.array( [ -1==+1,                                       1==+1 ] )
correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_log_likelihood           =', correct_ll
print 'output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients)

from math import sqrt

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix,coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors,feature_matrix[:,j])
            
            # add the step size times the derivative to the current coefficient
            ## YOUR CODE HERE
            coefficients[j] = coefficients[j] + step_size * derivative        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' %(int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)
print coefficients

# Compute the scores as a dot product between feature_matrix and coefficients.
scores = np.dot(feature_matrix, coefficients)
print scores

prob = predict_probability(feature_matrix,coefficients)
print prob
postive_review = prob[prob>0.5]
len(postive_review)

conv = lambda x: +1 if x > 0.5 else -1
pred = np.vectorize(conv)
pred = pred(prob)
diff = pred - sentiment
correct = diff[diff==0]
correct = len(correct)
total = len(sentiment)

num_mistakes = total-correct # YOUR CODE HERE
accuracy = correct / total # YOUR CODE HERE
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)


