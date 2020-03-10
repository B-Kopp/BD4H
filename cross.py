import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	k_fold = KFold(k, random_state=RANDOM_STATE)
	acc_folds = []
	auc_folds = []
    
	for fold, (train,test) in enumerate(k_fold.split(X,Y)):
		pred = models_partc.logistic_regression_pred(X[train], Y[train], X[test])
		acc, auc_, precision, recall, f1score = models_partc.classification_metrics(pred,Y[test])
		acc_folds.append(acc)
		auc_folds.append(auc_)
    
	return mean(acc_folds), mean(auc_folds)


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	ss = ShuffleSplit(n_splits=iterNo, test_size=test_percent, random_state=RANDOM_STATE)
	acc_folds = []
	auc_folds = []
    
	for fold, (train,test) in enumerate(ss.split(X,Y)):
		pred = models_partc.logistic_regression_pred(X[train], Y[train], X[test])
		acc, auc_, precision, recall, f1score = models_partc.classification_metrics(pred,Y[test])
		acc_folds.append(acc)
		auc_folds.append(auc_)
        
	return mean(acc_folds), mean(auc_folds)


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

