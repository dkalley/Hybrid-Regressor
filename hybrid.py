import numpy as np
import pandas as pd
import math
import copy
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import *


class HybridRegressor(BaseEstimator):    
    def __init__(self, base_estimators=[], rules=[], k=35, label="",models=[],features_selected=[],n_models=1,bucket_sample=0.60176991):
        # Models to be used per bucket
        self.models = models

        # A list of indices in order of smallest error to largest error
        self.model_ids = []

        # Top k features selected per bucket
        self.features_selected = features_selected

        # A list of rules to split the data on
        self.rules = rules

        # The string label for the dataset
        self.label = label

        # The base estimators to choose from
        self.base_estimators = base_estimators

        # The number of features to select
        self.k = k

        # The number of models used to predict
        self.n_models = n_models

        # Weights for each buckets models
        self.weights = []

        # The percentage to same from other buckets
        self.bucket_sample = bucket_sample

    # Method to set a full set of rules 
    def set_rules(self, new_rules):
        if not isinstance(new_rules, list):
            print('set_rules requires a list as the passed arguement')
            return
        self.rules = new_rules.copy()

    # Method to add a single new rule
    def add_rule(self, new_rule):
        self.rules.append(new_rule)

    # Feature selection for k features
    def feature_selection(self, X, y, k):
        # Merge the X dataset with the labels y
        concat = [X, y]
        X_new = pd.concat(concat, axis=1)

        # Save the label header
        self.label = y.columns.tolist()[0]

        # Perform bucketing
        buckets = self.kd(X_new)
        
        # Loop over each bucket and perform feature seelction
        features = []
        for bucket in buckets:
            f = []

            # Perform correlation ranking
            df_corr = bucket.corr()[self.label].sort_values(ascending=False).head(k+1)
            top_corr_features = df_corr.index.tolist()
            top_corr_features.pop(0)
            f.append(top_corr_features)

            # Perform random forest ranking
            rf_model = RandomForestRegressor()
            rf_model.fit(bucket.drop(columns=['SalePrice']), bucket['SalePrice'].values.ravel())
            feat_importances = pd.Series(rf_model.feature_importances_,index=X.columns)
            df_imp_feat = feat_importances.nlargest(k)
            f.append(df_imp_feat.index.tolist())

            # Perform weighted ranking and select weighted features
            features.append(self.weighted(f,k))

        # Set the features selected 
        self.features_selected = features

        # Return the list of selected features
        return features

    # Method to create weighted feature selection
    def weighted(self, list_features, k):
        weighted = {}

        # For a feature set in the list of features
        for feats in list_features:
            # Loop over all features
            for i in range(k-1):
                # If the feature is in the weighted dictionary then att the rank to its value
                if feats[i] in weighted:
                    weighted[feats[i]] = weighted[feats[i]] + i + 1
                # Otherwise create an element for the feature with the given rank
                else:
                    weighted[feats[i]] = i + 1

        # Sort the dictionary in order of smallest rank to largest
        sorted_weight = [k for k,v in sorted(weighted.items(), key=lambda item: item[1])]

        # Return the k best features
        return sorted_weight[:k]

    # Knowledge driven step splits data based on rules
    # Only works properly on numeric data
    def kd(self, X):
        dataset = []
        
        # If no rules are set return the whole dataset
        if self.rules == []:
            return [X]

        # Otherwise split on the rules
        for (key, buckets) in self.rules:
            temp = X.copy()
            for bucket in buckets:
                if bucket[0] != -1:
                    df = X[X[key].isin(bucket)].copy()
                else:
                    df = X[(X[key] > bucket[1]) & (X[key] <= bucket[2])].copy()
                
                dataset.append(df.copy())
        return dataset


    # Data Driven step trains models for each pipeline
    def dd(self, buckets):
        # Loop over all buckets or pipelines
        models_for_weight_generation = []
        new_buckets = []
        for idx, bucket in enumerate(buckets):
            # Generate buckets with sampling
            # Take 100% from original bucket and bucket_sample from all other buckets
            bs = []
            for i in range(len(buckets)):
                if i == idx:
                    bs.append(buckets[i].sample(frac=1))#n=272,replace=True))
                else:
                    bs.append(buckets[i].sample(frac=self.bucket_sample))#n=60,replace=True)) # Original 0.475
            bs_all = pd.concat(bs)
            y = bs_all[self.label].copy()
            X = bs_all[self.features_selected[idx]].copy()

            # Create combined buckets with feature selected
            new_bucket = X.copy()
            new_bucket[self.label] = y.copy()
            new_buckets.append(new_bucket)

            # If the base_estimators is empty then default to gradientboosting
            if self.base_estimators == []:
                self.base_estimators.append(GradientBoostingRegressor())
  
            # Loop over all base estimators 
            # Preform cross validation and choose the top models that gives the smallest error 
            results = []
            for estimator in self.base_estimators:
                results.append(cross_val_score(estimator, X, y.values.ravel(), scoring=make_scorer(root_mean_square)).mean())
            
            self.model_ids.append(results.index(min(results)))

            # Sort the models
            sort = sorted(range(len(results)), key=lambda k: results[k])

            # Append to the class attribute holding all the models for each pipeline
            m = []
            m2 = []
            for i in range(self.n_models):
                reg = copy.deepcopy(self.base_estimators[sort[i]])
                m.append(reg.fit(X,y.values.ravel()))
                m2.append(copy.deepcopy(self.base_estimators[sort[i]]))

            self.models.append(m)
            models_for_weight_generation.append(m2)

        # Generate the weights for the all the pipelines
        self.generate_weights(new_buckets, models_for_weight_generation)
        return
    
    # Method to generate weights to ensemble a pipeline's models
    def generate_weights(self, buckets, all_models):
        # List of lists
        # Each list is a buckets weights
        weights = []
        errors = []

        # Generate weights for each bucket
        for bucket,models in zip(buckets,all_models):
            # Create X and y for the given bucekt
            y = bucket[self.label].copy()
            X = bucket.drop(columns=[self.label]).copy()

            # Split the bucket into a test and train set
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

            # Train the models for this bucket
            my_testing_models = []
            for model in models:
                reg = model.fit(X_train, y_train)
                my_testing_models.append(reg)

            # Initialize weights here
            bucket_weights = [1/self.n_models for i in range(self.n_models)]

            # Loop until your criteria is met
            w = [1/self.n_models for i in range(self.n_models)]
            for i in range(1):
                predictions = []
                for model in my_testing_models:
                    predictions.append(model.predict(X_test))
                    
            errors.append(get_errors(predictions, y_test.values.ravel()))
        bucket_weights = auto_gen_weightsV2(errors)
        
        weights = bucket_weights

        # Set a class variable equal to the generated weights
        self.weights = weights

        return
        
    # Method to fit the hybrid model
    def fit(self, X, y):
        # Merge data and labels
        concat = [X, y]
        X_new = pd.concat(concat, axis=1)

        # If the feature label hasn't been saved
        if self.label == "":
            self.label = y.columns.tolist()[0]

        # Peform feature_selection
        self.feature_selection(X, y, self.k)

        # Perform knowledge and data driven stages
        buckets = self.kd(X_new)
        self.dd(buckets)

        # Return the model
        return self

    # Method to get the index for the model/features selected
    def get_index(self, row):            
        rule_num = 0
        count = 0

        # If no rules are set then there is only 1 model
        if self.rules == []:
            return (rule_num, count) 

        # Otherwise loop over all rules
        for (key, buckets) in self.rules:
            count = 0
            for bucket in buckets:
                if bucket[0] != -1:
                    if row[key] in bucket:
                        return (rule_num, count)
                    else:
                        count = count + 1
                else:
                    if (row[key] > bucket[1]) and (row[key] <= bucket[2]):
                        return (rule_num, count)
                    else:
                        count = count + 1
            rule_num = rule_num + 1
        
        # Print a row if its not found in a bucket
        print(row)  
        return          
    
    # Method to predict 
    def predict(self, X):
        results = []
        w_results = []

        # Iterate over the dataset row by row
        for idx, row in X.iterrows():
            (rule_num, count) = self.get_index(row)
            i = 0
            # If there are a set of rules get the correct i
            if self.rules != []:
                for i in range(rule_num):
                    i = i + len(self.rules[i][1]) - 1
                i = i + count

            # Predict on every bucket and merge
            r = 0
            for j in range(len(self.rules[0][1])):
                if i == j:
                    r = r + 0.60176991*self.models[j][0].predict([row[self.features_selected[j]]])
                else:
                    r = r + 0.13274336*self.models[j][0].predict([row[self.features_selected[j]]])

        return np.array(results)

    # Method to predict by ensembling all models in each pipeline
    def predict_weighted(self, X):
        results = []
        w_results = []

        # Iterate over the dataset row by row
        for idx, row in X.iterrows():
            (rule_num, count) = self.get_index(row)
            i = 0
            # If there are a set of rules get the correct i
            if self.rules != []:
                for i in range(rule_num):
                    i = i + len(self.rules[i][1]) - 1
                i = i + count

            # Predict using the correct model and features
            row_result = []
            for model in self.models[i]:
                row_result.append(math.floor(model.predict([row[self.features_selected[i]]])))

            # Apply weights here
            pred = 0
            for w,p in zip(self.weights[i],row_result):
                pred = pred + w*p
            w_results.append(pred)
        
        return np.array(w_results)

    # Method to return a score from a prediction and actual
    def score(self, prediction, actual):
        return root_mean_square(actual, prediction)

    def get_params(self, deep=True):
        return {"base_estimators": self.base_estimators, "rules": self.rules, "k":self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Method for scoring
def root_mean_square(actual, pred):
    # Log predictions and actual
    y_actual = np.log(actual)
    y_pred = np.log(np.absolute(pred))

    # Perform root mean square error
    mse = np.square(np.subtract(y_actual,y_pred)).mean()
    return math.sqrt(mse)

# Cross validation method 
def cross_validate(reg, X,y ,scoring=root_mean_square,k_fold=5):
    results = []
    for i in range(k_fold):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        reg_new = copy.deepcopy(reg)
        # Fit, predict, save results
        reg_new.fit(X_train, y_train)
        predictions = reg_new.predict(X_test)
        results.append(reg_new.score(predictions, y_test.values.ravel()))
    
    return np.array(results)

# Cross validation method for using all models in a given pipeline
def cross_validate_weighted(reg, X,y ,scoring=root_mean_square,k_fold=5):
    results = []
    for i in range(k_fold):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        reg_new = copy.deepcopy(reg)
        # Fit, predict, save results
        reg_new.fit(X_train, y_train)
        predictions = reg_new.predict_weighted(X_test)
        results.append(reg_new.score(predictions, y_test.values.ravel()))
    
    return np.array(results)

# Method to produce errors from a list of predictions
def get_errors(pred, labels):
    
    errors = []
    error = 0
    for prediction in pred:
        errors.append(root_mean_square(labels, prediction))
    return errors
    
# Final version of a method to generate pipeline weights
def auto_gen_weightsV2(errors):
    # Loop over each pipeline
    final_weights = []
    for bucket in range(len(errors)):
        # Sort the errors of each pipeline's models
        ranks = [i for i in errors[bucket]]

        # Produce a ranking of errors
        ranks_copy = ranks.copy()
        for i in range(len(errors[bucket])):
            index = ranks_copy.index(max(ranks_copy))
            ranks[index] = len(ranks) - i
            
            ranks_copy[index] = 0

        er_copy = ranks.copy()
        er_copy.sort(reverse=False)
        
        # Generate the weight
        m = er_copy[0]
        for i in range(len(er_copy)):
            er_copy[i] = (er_copy[i] - m) + 1 * (i+1)

        s = sum(er_copy)
        
        # Normalize the weights
        for i in range(len(er_copy)):
            er_copy[i] = er_copy[i] / s
        
        # Store the weights
        weights = er_copy.copy()
        for i in range(len(ranks)):
            index = ranks.index(i+1)
            weights[index] = er_copy[(len(ranks) - 1 - i)]
        
        final_weights.append(weights)
    return final_weights
    