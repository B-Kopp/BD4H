import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file

import utils

RANDOM_STATE=1215
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	X_train, Y_train = utils.get_data_from_svmlight("deliverables/features_svmlight.train")

	filepath = 'data/test/'

	#Columns in events.csv - patient_id,event_id,event_description,timestamp,value
	events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in event_feature_map.csv - idx,event_id
	feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

	#Aggregate Events of test features
	aggregated_events = pd.merge(events, feature_map, how='inner', on='event_id')
	aggregated_events = aggregated_events.dropna(subset=['value'])
	aggregated_events = aggregated_events[['patient_id', 'event_id', 'value', 'idx']].groupby(['patient_id','event_id', 'idx']).agg('count').reset_index()
	grouper = aggregated_events.groupby('idx')['value']
	maxes = grouper.transform('max')
	aggregated_events = aggregated_events.assign(value=(aggregated_events.value/maxes))
	aggregated_events = aggregated_events.rename(columns={'idx': 'feature_id', 'value': 'feature_value'})
	aggregated_events = aggregated_events.drop(columns=['event_id'])

	#Create test features
	aggregated_events = aggregated_events.astype({'patient_id': float, 'feature_id': float, 'feature_value': float})
	patient_features = aggregated_events.set_index('patient_id').apply(tuple,1).groupby(level=0).agg(lambda x: list(x.values)).to_dict()

	#Save test features
	deliverable = open('deliverables/test_features.txt', 'wb')

	for k, v in patient_features.items():
		deliverable.write(bytes((str(int(k)) + ' '),'UTF-8'));
		v.sort(key=lambda tup: tup[0])
		for p in v:
			deliverable.write(bytes((str(int(p[0]))+':'+f'{p[1]:.6f}'+' '),'UTF-8'));
		deliverable.write(bytes((u'\n'),'UTF-8'));

	X_test = load_svmlight_file('deliverables/test_features.txt', n_features=3190)
	X_test = X_test[0]

	return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	et_model = ExtraTreesClassifier(n_estimators=1300, min_samples_split=0.0001, max_depth=70, n_jobs=-1, random_state=RANDOM_STATE)
	et_model.fit(X_train,Y_train)
	Y_pred1 = et_model.predict_proba(X_test)[:,1]
	svm_model = SVC(C=150, probability=True, random_state=RANDOM_STATE)
	svm_model.fit(X_train, Y_train)
	Y_pred3 = svm_model.predict_proba(X_test)[:,1]

	Y_pred = 0.5*Y_pred1 + 0.5*Y_pred3

	return np.round(Y_pred).astype(int)


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	