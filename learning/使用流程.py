# 本脚本只要是记录如何使用这几个工具文件的

# 0. 数据准备部分。
# 在数据处理的最后使用numpy的保存功能将数据保存为npy格式备用


# 1。Keras部分
# 使用keras_hyperas.py（拷贝到notebook里面），并调整初次的模型网络。 这里输出hyperas得到的最优参数；
# 根据得到的最优参数先构建一个mlp模型。（无需加入model.compile)
# 使用keras_tunning_lr.py作图并获得最优optimizer
import keras_tunning_lr as tlr
x_train, y_train, x_test, y_test = data()
data_set = [x_train, x_test, y_train, y_test]
tunning_lr = tlr.LR_Tunning(model, batch_size=128, epochs=20, dataSet=data_set)


# 得到最优optimizer后，可以使用wrappers构造mlp的模型备用，如果不用wrapper的话后面很难跟其他sklearn的模型进行融合。
from keras.wrappers.scikit_learn import KerasClassifier
mlp_model = kerasClassifier(build_fn=create_model, epochs=20, batch_size=128)



# 2. sklearn部分
# 使用 sklearn_tunning 进行sklearn相关模型的参数优化。
# 目前进行参数优化的模型有RF，XGB和LGBM
# 输出的模型有["Naive Bayes", "Decision Tree", "RF", "GDBT", "ADBT", "XGB", "LGBM", "SVM"]， 注意这里svm的运行速度是相当慢的， 初期可以移除
import sklearn_tunning as skt
X = np.load('')
Y = np.load('')
sktune = skt.SK_Tunning(X,Y)

models = sktune.run()


# 3.融合部分
# 为增加模型的多样性，除了上阶段的几个模型外，可以额外的增加新模型进行融合。（增加模型前可以先用上面的几个模型先跑一版）
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

clf1 = SGDClassifier(
            alpha=5e-05,
            average=False,
            class_weight='balanced',
            loss='log',
            n_iter=30,
            penalty='l2', n_jobs=-1, random_state=random_rate)
			
clf2 = LinearSVC(C=0.1, random_state=random_rate)
clf2 = CalibratedClassifierCV(base_estimator=clf2) # 这里因为LinearSVC不能预测概率，需要包裹一层

clf3 = LogisticRegression(C=1.0,n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate)
clf4 = BernoulliNB(alpha=0.1)

clf5 = SGDClassifier(
		alpha=5e-05,
		average=False,
		class_weight='balanced',
		loss='log',
		n_iter=30,
		penalty='l1', n_jobs=-1, random_state=random_rate)
		
clf6 = LinearSVC(C=0.9, random_state=random_rate)
clf6 = CalibratedClassifierCV(base_estimator=clf6)

clf7 = LogisticRegression(C=0.5, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate)

clf8 = BernoulliNB(alpha=0.9)
clf9 = LogisticRegression(C=0.2, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,penalty='l1')
clf10 = LogisticRegression(C=0.8, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,penalty='l1')

clf11 = LinearSVC(C=0.5, random_state=random_rate)
clf11 = CalibratedClassifierCV(base_estimator=clf2)

clf12 = BernoulliNB(alpha=0.5)

basemodels = [clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11,clf12]
	
# 用法
import stacking as st
s = st.Stacking()


# 目前接受的输入时dataframe格式
pred_st = s.fit_multi_lgb(X=X, y=Y, X_target=test_x, models=final_models)

# 评判方式
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

print (recall_score(y_pred=pred_st, y_ture=test_y))
print (precision_score(y_pred=pred_st, y_ture=test_y))
print (f1_score(y_pred=pred_st, y_ture=test_y))

confusion_matrix(y_pred=pred_st, y_ture=test_y)