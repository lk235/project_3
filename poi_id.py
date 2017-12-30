#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import Counter
import matplotlib.pyplot
from tester import *
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## Original features_list
# features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
#                  'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
#                  'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
#                  'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print len(data_dict)
print 'features_count: ',len(data_dict['METTS MARK'])
poi_count = 0
non_poi_count = 0
nan_list = []
for key,value in data_dict.iteritems():
    for k,v in value.iteritems():
        if k != 'email_address' and v == 'NaN':
            nan_list.append(k)
            #Repalce 'NaN' with 0
            data_dict[key][k] = 0

    if value['poi'] == True:
        print key
        poi_count = poi_count + 1
    else:
        non_poi_count = non_poi_count + 1

print 'poi count: ',poi_count
print 'non_poi_count: ',non_poi_count
print 'feature_nan_list: ',Counter(nan_list)

#Choose two features as a scatter plot
features = ["total_payments", "total_stock_value"]

# Remove 'TOTAL' item
data_dict.pop('TOTAL')

data = featureFormat(data_dict, features)
for point in data:
    total_payments= point[0]
    total_stock_value = point[1]
    matplotlib.pyplot.scatter( total_payments, total_stock_value )

matplotlib.pyplot.xlabel("total_payments")
matplotlib.pyplot.ylabel("total_stock_value")
# matplotlib.pyplot.show()

for key,value in data_dict.iteritems():
     if value['total_payments'] > 100000000 and value['total_payments'] != 'NaN':
         print 'Outlier :',key


### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    if poi_messages == 0 or all_messages == 0:
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)

    return fraction

for name in data_dict:

    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi


## features_list with new features
# features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
#                  'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
#                  'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
#                  'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi']

# features_list with deleted features
features_list = ['poi','salary','total_payments', 'bonus',
                 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'restricted_stock',
                 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi']



### Store to my_dataset for easy export below.
## dataset without reduce
# my_dataset = data_dict

## dataset with reduce
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np
df = pd.DataFrame.from_dict(data_dict,orient='index')
scaler = MinMaxScaler()
for feature in features_list:
    if feature != 'email_address'and feature != 'poi':
        df[feature] = scaler.fit_transform(df[feature])

my_dataset = df.to_dict(orient='index')



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# TEST 3 classfier without tuuning and with original features_list
clf = GaussianNB()
dump_classifier_and_data(clf,my_dataset,features_list)
print main()

clf = SVC(kernel='linear')
dump_classifier_and_data(clf,my_dataset,features_list)
print main()

clf = DecisionTreeClassifier()
dump_classifier_and_data(clf,my_dataset,features_list)
print main()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

# Use feature_importances_ to select features
clf = DecisionTreeClassifier()
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

    ### fit the classifier using training set, and test on test set

    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)

feature_importances = clf.feature_importances_

#Display the feature names and importance values
tree_important_features = []
tree_important_features_list = []

for feature in zip(sorted(feature_importances,reverse=True), features_list):
    if feature[0] > 0.1:
        tree_important_features.append(feature)
        tree_important_features_list.append(feature[1])
print 'tree_important_features :',tree_important_features
print 'tree_important_features_list :',tree_important_features_list
# features_list = tree_important_features_list

# Use SelectBest to select features
selector = SelectKBest(k=4).fit(features_train, labels_train)

def get_new_features(selector,features_list):
    new_features = []
    for bool, feature in zip(selector.get_support(), features_list):
        if bool:
            new_features.append(feature)
    return new_features

print 'selectBEST :',get_new_features(selector,features_list)
features_list = get_new_features(selector,features_list)

# parameter tuning
parameters = {'min_samples_split': [2,5,10,20,30],'max_depth': range(1,5),'class_weight':[None,'balanced']}
cv = StratifiedShuffleSplit(labels,1000,random_state=18)
clf_tree = GridSearchCV(DecisionTreeClassifier(),parameters,cv=10,scoring='f1')
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
clf_tree.fit(features,labels)

print clf_tree.best_params_
print clf_tree.best_estimator_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(min_samples_split= 20 , max_depth= 3,class_weight = 'balanced')

dump_classifier_and_data(clf,my_dataset,features_list)
print main()
print features_list


