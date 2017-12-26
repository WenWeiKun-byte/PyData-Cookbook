import numpy as np
import pandas as pd

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve,accuracy_score, precision_score, recall_score,f1_score

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# 扩展同时支持多个分数输出
from multiscorer import MultiScorer  #https://github.com/StKyr/multiscorer/
import time

"""
对rf，xgb，lgb关键参数调参并输出建议参数

https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt

目前支持F1，AUC。因为fmin是查找满足结果最小的参数，所以在建立scorer的时候需要把目标转换成负数.
"""

class SK_Tunning():
	def __init__(self, X, Y):

		self.X = X
		self.Y = Y
		self.n_splits = 6
		self.max_evals = 10

	
	def run(self):
		X = self.X
		Y = self.Y
		# 分数部分
		def f1_score_(truth, predictions):
			return -1* f1_score(y_true=truth, y_pred=predictions) # 其实就是转换成负数让他能应用于fmin
			
		f1_scorer = make_scorer(f1_score_, greater_is_better=True)


		# auc 分数
		# auc 不是内置的scorer，而且需要用到结果的概率，所以需要自己计算相关
		def get_auc_score(clf,X,Y,n_splits):
			cv = StratifiedKFold(n_splits=n_splits)
			cv_probs = cross_val_predict(clf, X, Y, method='predict_proba', cv=cv)[:,1]
			auc = []
			for train_idx, test_idx in cv.split(X, Y):
				auc.append(roc_auc_score(Y[test_idx], cv_probs[test_idx]))
			return np.mean(auc) * -1


		# RF部分
		def rf_objective(params):
			params = {'n_estimators': int(params['n_estimators']), 
					 'max_depth': int(params['max_depth'])}
			clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)
			# score = cross_val_score(clf, X, Y, scoring=f1_scorer, cv=StratifiedKFold(n_splits=self.n_splits)).mean()
			score = get_auc_score(clf, X, Y, self.n_splits)
				
			print ('AUC {:.5f} params {}'.format(score, params))
			return score

		rf_space = {
			'n_estimators': hp.quniform('n_estimators', 25, 800, 25),
			'max_depth': hp.quniform('max_depth', 1, 15, 1)
		}
		print ('exploring parameters for RF')
		rf_best = fmin(fn=rf_objective,
				   space=rf_space,
				   algo=tpe.suggest,
				   max_evals=self.max_evals)
				   
		# xgb部分
		
		def xgb_objective(params):
			params = {'gamma': "{:.3f}".format(params['gamma']), 
					 'max_depth': int(params['max_depth']),
					 'colsample_bytree': "{:.3f}".format(params['colsample_bytree'])}
			clf = xgb.XGBClassifier(n_jobs=4, 
									n_estimators=250,
									learning_rate=0.05,
									**params)
		#     score = cross_val_score(clf, X, Y, scoring=f1_scorer, cv=StratifiedKFold(n_splits=6)).mean()
			score = get_auc_score(clf, X, Y, self.n_splits)
				
			print ('AUC {:.5f} params {}'.format(score, params))
			return score

		xgb_space = {
			'gamma': hp.uniform('gamma', 0.0, 0.5),
			'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
			'max_depth': hp.quniform('max_depth', 1, 15, 1)
		}
		
		print ('exploring parameters for XGB')
		xgb_best = fmin(fn=xgb_objective,
				   space=xgb_space,
				   algo=tpe.suggest,
				   max_evals=self.max_evals)
				   
				   
		def lgbm_objective(params):
			params = { 
					 'num_leaves': int(params['num_leaves']),
					 'colsample_bytree': "{:.3f}".format(params['colsample_bytree'])}
			clf = lgb.LGBMClassifier(
									n_estimators=500,
									learning_rate=0.01,
									**params)
		#     score = cross_val_score(clf, X, Y, scoring=f1_scorer, cv=StratifiedKFold(n_splits=6)).mean()
			score = get_auc_score(clf, X, Y, self.n_splits)
				
			print ('AUC {:.5f} params {}'.format(score, params))
			return score

		lgbm_space = {
			
			'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
			'num_leaves': hp.quniform('num_leaves', 8, 128, 2)
		}
		print ('exploring parameters for LGBM')
		lgbm_best = fmin(fn=lgbm_objective,
				   space=lgbm_space,
				   algo=tpe.suggest,
				   max_evals=self.max_evals)
				   
				   
		#全模型比较
		rf_params = {'n_estimators': int(rf_best['n_estimators']), 
					 'max_depth': int(rf_best['max_depth'])}
		rf_model = RandomForestClassifier(n_jobs=4, class_weight='balanced', **rf_params)
		
		xgb_params  = {'gamma': "{:.3f}".format(xgb_best['gamma']), 
					 'max_depth': int(xgb_best['max_depth']),
					 'colsample_bytree': "{:.3f}".format(xgb_best['colsample_bytree'])}
		xgb_model = xgb.XGBClassifier(n_jobs=4, 
									n_estimators=250,
									learning_rate=0.05,
									**xgb_params)
									
		lgbm_params = { 
					 'num_leaves': int(lgbm_best['num_leaves']),
					 'colsample_bytree': "{:.3f}".format(lgbm_best['colsample_bytree'])}						
		lgbm_model = lgb.LGBMClassifier(
									n_estimators=500,
									learning_rate=0.01,
									**lgbm_params)
									
		models = [GaussianNB(), DecisionTreeClassifier(), rf_model ,GradientBoostingClassifier(), AdaBoostClassifier(),xgb_model,lgbm_model, SVC(probability=True)]
		names = ["Naive Bayes", "Decision Tree", "RF", "GDBT", "ADBT", "XGB", "LGBM", "SVM"]

		scorer = MultiScorer({
			'Accuracy' : (accuracy_score, {}),
			'Precision' : (precision_score, {}),
			'Recall' : (recall_score, {}),
			'F1' : (f1_score, {})
		})

		for model, name in zip(models, names):
			print (name)
			start = time.time()

			cross_val_score(model, X, Y, scoring=scorer, cv=10)
			results = scorer.get_results()

			for metric_name in results.keys():
				average_score = np.average(results[metric_name])
				scores = np.array(results[metric_name])
				print("%s: %0.5f (+/- %0.3f)" % (metric_name, scores.mean(), scores.std() * 2))
		#         print('%s : %f' % (metric_name, average_score))

			print ('time', time.time() - start, '\n\n')
		return models