import pandas as pd
import numpy as np
import math
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.kernel_ridge import KernelRidge

# My hybrid file. Just needs to be in the same directory
from hybrid import HybridRegressor
from hybrid import cross_validate
from hybrid import cross_validate_weighted
from hybrid import root_mean_square

##############################################################################################
#   Load datasets
##############################################################################################
# Load train set and fill any na entries
train_set = pd.read_csv('train.csv')
train_set.fillna(0,inplace=True)

# Create X and y from the train set
y = train_set[['SalePrice']].copy()
X = train_set.drop(columns=['SalePrice'])

# Load test set and fill any na entries
test_set = pd.read_csv('test.csv')
test_set.fillna(0,inplace=True)


##############################################################################################
#   Split Train set for testing
##############################################################################################
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

##############################################################################################
#   Create rule sets (Optional)                     
#   Make sure every data point is covered by a rule
#   Template for a rule: (tuple)
#       (Feature rule split on, list of buckets where each bucket shows all values within the bucket)
##############################################################################################
# Should be a list of tuples
rules = []
# 4 - Buckets Average house price ranges [0,145000) [145000,165000) [165000, 225000) [225000,315000]
rules.append(('Neighborhood',[[2, 3, 4, 8, 10, 11, 15, 18, 19, 20],[12, 13],[1, 5, 6, 7, 9, 17, 21],[14, 16, 22, 23, 24, 25]]))

##############################################################################################
#   Create set of n models otherwise a base GradientBoostingRegressor() will be used
##############################################################################################
rf = RandomForestRegressor()
rf1 = RandomForestRegressor(max_depth=3)
ens = VotingRegressor(estimators=[('rf',rf),('rf1',rf1)])
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber',random_state=1)
gb1 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber',random_state=1)
gb2 = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber',random_state=1)
gb3 = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber',random_state=1)
gb4 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber',random_state=1)
gb_ens = VotingRegressor(estimators=[('gb',gb),('gb1',gb1), ('gb2',gb2),('gb3',gb3),('gb4',gb4)])
all_ens = VotingRegressor(estimators=[('rf_ens', ens),('gb_ens',gb_ens)])
estimators = [('', RandomForestRegressor(n_estimators=300,random_state=1)),
                ('kernel_ridge', KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)),
              ('Boosting', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber',random_state=1)),
              ('elasticnet', ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)),
              ( 'lasso', Lasso(alpha =0.0005, random_state=1)),
             ]
stack_reg = StackingRegressor(estimators = estimators, final_estimator = RandomForestRegressor(n_estimators = 500 ,random_state=1), n_jobs=-1)
sr = StackingRegressor(estimators=([('rf_ens', ens),('gb_ens',gb_ens)]), final_estimator=rf)

# Create a list of model names and another list of all the models
labels = ['gb','rf1','gb4','ens','gb_ens','all_ens']
models = [gb,rf1,gb4,ens,gb_ens, all_ens]

##############################################################################################
#   Either test or predict with model
##############################################################################################

# By setting testing to True, the script will use the training data to generate experimental results
# Otherwise, Prediciting will be true and the script will generate a list of predictions based on the test set
TESTING = False
PREDICTING = not TESTING

# Set the number of features to select
k = 35

if TESTING:
    # Call hybrid regressor passing base_estimators as the set you created
    reg = HybridRegressor(base_estimators=models, rules=rules, k=k, n_models=3)

    '''
    Code to cross validate data and print the outputs in a meaningful way
    # Cross Validate
    results = cross_validate(reg, X, y)

    print('Run 1   | Run 2   | Run 3   | Run 4   | Run 5   | std     | mean')
    print('===================================================================')
    row = ''
    for result in results:
        print('{:.5f} | '.format(result),end='')
    print('{:.5f} | {:.5f} '.format(np.array(result).std(), np.array(results).mean()))
    '''

    # Fit with a sample of the train set
    reg.fit(X_train, y_train)

    # Predict for both versions of the hybrid regressor
    predictions = reg.predict(X_test)
    weighted_pred = reg.predict_weighted(X_test)

    # Print the error when only using 1 model per pipeline and ensembling all pipelines
    base_hybrid = reg.score(predictions, y_test.values.ravel())
    print('\tHybrid Error   : {}'.format(base_hybrid))

    # Print the error when using 3 model per pipeline and not ensembling pipelines
    bucket_ensemble = reg.score(weighted_pred, y_test.values.ravel())
    print('\tFinal Pipeline Ensembling Error: {}'.format(bucket_ensemble))
    

if PREDICTING:
    # Call hybrid regressor passing base_estimators as the set you created
    reg = HybridRegressor(base_estimators=models, rules=rules, k=k, n_models=3, bucket_sample=0.6)

    # Fit with the train set
    reg.fit(X, y)

    # Predict with the test set
    predictions = reg.predict(test_set)
    predictions_weighted = reg.predict_weighted(test_set)

    print('Hybrid Predictions\n',predictions)
    print('Hybrid with Pipeline Ensembling Predictions\n',predictions_weighted)