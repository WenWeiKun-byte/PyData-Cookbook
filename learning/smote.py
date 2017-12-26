import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# get the source data
data = pd.read_csv('../datasource/creditcard.csv')

ss = StandardScaler()
data['time_'] = ss.fit_transform(data.Time.values.reshape(-1, 1))
data['amount_'] = ss.fit_transform(data.Amount.values.reshape(-1, 1))
del data['Time']
del data['Amount']
sub_non_fraud = data.loc[data.loc[:, 'Class']===0,:].sample(int(data.shape[0] / 2))
sub_fraud = data.loc[data.Class==1,:]
data_resample = pd.concat([sub_fraud, sub_non_fraud])

X = data_resample.loc[:, data_resample.columns != 'Class']
y = data_resample.loc[:, data_resample.columns == 'Class']

# 应该在smote之前就把测试和建模的数据分好，因为后面的数据都是经过smote处理的，
# 你很难保证你的模型预测的提高是因为预测到了smote生成的负样本还是真实的负样本。

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) #这里stratify保证y的01分布与源数据一致

sm = SMOTE(kind='regular')
X_resampled, y_resampled = sm.fit_sample(x_train, y_train)
np.save('sm_x', X_resampled)
np.save('sm_y', y_resampled)
np.save('or_x', x_test.values)
np.save('or_y', y_test.values)