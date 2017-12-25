import pandas as pd
import numpy as np

import copy

from lightgbm import LGBMClassifier

"""
usage:

import stacking as st
s = st.Stacking()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)  #需要在外部对数据先切分好。
pred_LGB = s.fit_multi_lgb(
        X=x_train,
        y=y_train,
        X_target=x_test,
        models=models
    )  # models是上一步 sklearn_tunning得出的n个调参后的模型。返回的是融合模型的预测结果。
	
# 通过pred_LGB与y_test的结果比对，得出模型融合后的相关评判分数：
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,f1_score

accuracy_score(y_true=y_test, y_pred=pred_LGB)
precision_score(y_true=y_test, y_pred=pred_LGB)
recall_score(y_true=y_test, y_pred=pred_LGB)
f1_score(y_true=y_test, y_pred=pred_LGB)
"""

class Stacking():
    def __init__(self):
        self.k_fold = 3
        pass
    def split_data(self, data, num_parts):
        result = []
        length = len(data)
        for i in range(num_parts):
            start = length * i//num_parts
            end   = length * (i+1)//num_parts
            result.append(data[start:end])
        return result
    def fit(self, X, y, X_target, model):
        X_split = self.split_data(X, self.k_fold)
        y_split = self.split_data(y, self.k_fold)
        result = []
        for i in range(self.k_fold):
            print('第'+str(i)+'次迭代')
            X_train = copy.deepcopy(X_split)
            y_train = copy.deepcopy(y_split)
            X_train.pop(i)
            y_train.pop(i)
            all_features = X_train[0]
            if len(X_train) > 0:
                for feature in X_train[1:]:
                    all_features = pd.concat([all_features, feature], axis=0)
            X_train = all_features
            all_features = y_train[0]
            if len(y_train) > 0:
                for feature in y_train[1:]:
                    all_features += feature
            y_train = all_features
            # X_test = X_split[i]
            # y_test = y_split[i]
            model.fit(X_train, y_train)
            predictions = model.predict(X_target)
            result.append(predictions)
        pred_result = []
        for i in range(X_target.shape[0]):
            preds_from_k_model = [r[i] for r in result]
            pred_result.append(np.average(preds_from_k_model, axis=0))
        return pred_result
    # 使用两层LGB进行预测
    def fit_double_lgb(self, X, y, X_target):
        X_split = self.split_data(X, self.k_fold)
        y_split = self.split_data(y, self.k_fold)
        train_result = []
        test_result = []
        for i in range(self.k_fold):
            # 分割训练集
            print('第'+str(i+1)+'次迭代########################################################')
            X_train = copy.deepcopy(X_split)
            y_train = copy.deepcopy(y_split)
            X_train.pop(i)
            y_train.pop(i)
            all_features = X_train[0]
            if len(X_train) > 0:
                for feature in X_train[1:]:
                    all_features = pd.concat([all_features, feature], axis=0)
            X_train = all_features
            all_features = y_train[0]
            if len(y_train) > 0:
                for feature in y_train[1:]:
                    all_features += feature
            y_train = all_features
            # 用分割的训练集训练模型
            lgb_model = LGBMClassifier()
            lgb_model.fit(X_train, y_train)
            pred_train = lgb_model.predict(X)
            pred_test = lgb_model.predict(X_target)
            pred_train = [np.argmax(pred) for pred in pred_train]
            pred_test = [np.argmax(pred) for pred in pred_test]
            train_result.append(pred_train)
            test_result.append(pred_test)
        # 获得train和test集的新表示，并进行第二次lgb训练
        new_train_representation = []
        for index in range(X.shape[0]):
            new_train_representation.append([vec[index] for vec in train_result])
        new_test_representation = []
        for index in range(X_target.shape[0]):
            new_test_representation.append([vec[index] for vec in test_result])
        new_train_representation = np.array(new_train_representation, dtype=np.float64)
        # 开始最终的预测
        print('Start final Predicting ... ')
        lgb_model = LGBMClassifier()
        lgb_model.fit(new_train_representation, y)
        preds = lgb_model.predict(new_test_representation)
        return preds
    # 使用多个模型进行预测
    def fit_multi_lgb(self, X, y, X_target, models):
        X.columns = range(X.shape[1])
        X_target.columns = range(X_target.shape[1])
        X_split = self.split_data(X, self.k_fold)
        y_split = self.split_data(y, self.k_fold)
        all_train_features = pd.DataFrame([])
        all_test_features = pd.DataFrame([])
        for index, model in enumerate(models):
            cur_model = copy.deepcopy(model)
            # 使用全部数据进行预测
            cur_model.fit(X, y)
            # test_result = pd.DataFrame([np.argmax(vec) for vec in cur_model.predict(X_target)])
            test_result = pd.DataFrame(cur_model.predict_proba(X_target)[:, 1])
            # 构造K-折结果作为特征
            train_result = []
            for i in range(self.k_fold):
                # 分割训练集
                print('第'+str(index+1)+'个模型， 第' + str(i + 1) + '次迭代########################################################')
                X_train = copy.deepcopy(X_split)
                y_train = copy.deepcopy(y_split)
                X_train.pop(i) # 把其中一块移除
                y_train.pop(i)
                all_features = X_train[0]
                if len(X_train) > 0:
                    for feature in X_train[1:]: # 剩下的
                        all_features = pd.concat([all_features, feature], axis=0) # 相当于挖掉一块，把剩下的整合起来
                X_train = all_features
                X_train.columns = range(X_train.shape[1])
                all_features = y_train[0]
                if len(y_train) > 0:
                    for feature in y_train[1:]:
                        #all_features += feature
                        all_features = pd.concat([all_features, feature], axis=0)
                y_train = all_features
                # 用分割的训练集训练模型
                lgb_model = copy.deepcopy(model)
                lgb_model.fit(X_train.values, y_train.values.ravel())
                pred_train = lgb_model.predict_proba(X_split[i].values)[:, 1]
                # pred_train = [np.argmax(pred) for pred in pred_train]
                train_result.append(pd.DataFrame(pred_train))
            # train_result = pd.DataFrame(sum(train_result, []))
            train_result = pd.concat(train_result, axis=0)
            print (train_result)
            # 将新特征加入到总特征序列中去
            all_train_features = pd.concat([all_train_features, train_result], axis=1)
            all_test_features  = pd.concat([all_test_features, test_result], axis=1)
        print('Train Features Shape', all_train_features.shape)
        print('Test Features Shape', all_test_features.shape)
        lgb_overall = LGBMClassifier(n_estimators=500, learning_rate=0.01)
        lgb_overall.fit(all_train_features.values, y.values.ravel())
        preds = lgb_overall.predict(all_test_features.values)
        return preds