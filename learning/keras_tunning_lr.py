from __future__ import print_function
import numpy as np
import math
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.models import model_from_json
from keras.models import clone_model
from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt	

class LR_Tunning():
	def __init__(self, model, batch_size, epochs, dataSet):
		"""
		dataSet will be a list: [x_train, x_test, y_train, y_test]
		
		example:
		import keras_tunning_lr as tlr
		tunning_lr = tlr.LR_Tunning(model, batch_size=64, epochs=20, dataSet=data_set)
		tunning_lr.run()
		
		# 最后在img/下保存acc和loss的比较图
		"""
		self.model = model
		self.batch_size = batch_size
		self.epochs = epochs
		x_train, x_test, y_train, y_test = dataSet
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		self.y_test = y_test
		self.result = []
		
	def plot_fig(i, history):
		fig = plt.figure()
		plt.plot(range(1,self.epochs+1),history.history['val_acc'],label='validation')
		plt.plot(range(1,self.epochs+1),history.history['acc'],label='training')
		plt.legend(loc=0)
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		plt.xlim([1,self.epochs])
		# plt.ylim([0,1])
		plt.grid(True)
		plt.title("Model Accuracy")
		plt.show()
		# fig.savefig('img/'+str(i)+'-accuracy.jpg')
		plt.close(fig) 
			
	
		
	def run(self):
		X_train = self.x_train
		X_test = self.x_test
		y_train = self.y_train
		y_test = self.y_test
		batch_size = self.batch_size
		epochs = self.epochs
		
		
		# the default SGD
		# define SGD optimizer
		model1 = clone_model(self.model)
		learning_rate = 0.1
		sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False) # set to default except lr
		# compile the model
		model1.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])

		# fit the model
		history_default_sgd = model1.fit(X_train, y_train,
						batch_size=batch_size,
						epochs=epochs,
						verbose=2,
						validation_data=(X_test, y_test))
		
		
		# time base decay
		model2 = clone_model(self.model)
		learning_rate = 0.1
		decay_rate = learning_rate / self.epochs
		momentum = 0.5
		sgd_time_base = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
		# compile the model
		model2.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd_time_base, metrics=['accuracy'])

		# fit the model
		history_time_sgd = model2.fit(X_train, y_train, 
							 epochs=epochs, 
							 batch_size=batch_size,
							 verbose=2, 
							 validation_data=(X_test, y_test))
		
		#Step decay		
		def step_decay(epoch):
			initial_lrate = 0.1
			drop = 0.5
			epochs_drop = 10.0 
			lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
			return lrate
		
		model3 = clone_model(self.model)
		momentum = 0.5
		sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False)
		model3.compile(loss=keras.losses.binary_crossentropy,optimizer=sgd, metrics=['accuracy'])			 
		lrate = LearningRateScheduler(step_decay)
		callbacks_list = [lrate]
		history_step_sgd = model3.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     callbacks=callbacks_list, 
                     verbose=2)
		
		# Exponential decay
		def exp_decay(epoch):
			initial_lrate = 0.1
			k = 0.1
			lrate = initial_lrate * np.exp(-k*epoch)
			return lrate	
		model4 = clone_model(self.model)
		# define SGD optimizer
		momentum = 0.8
		sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False)

		# compile the model
		model4.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
		lrate_ = LearningRateScheduler(exp_decay)
		callbacks_list_ = [lrate_]
		history_exp_sgd = model4.fit(X_train, y_train, 
				 validation_data=(X_test, y_test), 
				 epochs=epochs, 
				 batch_size=batch_size, 
				 callbacks=callbacks_list_, 
				 verbose=2)
				 
		# 
		# using Adagrad optimizer
		model5 = clone_model(self.model)
		model5.compile(loss=keras.losses.binary_crossentropy,
					  optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
					  metrics=['accuracy'])
		history_agagrad = model5.fit(X_train, y_train, 
							 validation_data=(X_test, y_test), 
							 epochs=epochs, 
							 batch_size=batch_size,
							 verbose=2)
		
		# using Adadelta		
		model6 = clone_model(self.model)
		model6.compile(loss=keras.losses.binary_crossentropy,
					  optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
					  metrics=['accuracy'])
		history_adadelta = model6.fit(X_train, y_train, 
							 validation_data=(X_test, y_test), 
							 epochs=epochs, 
							 batch_size=batch_size,
							 verbose=2)
							 
		model7 = clone_model(self.model)
		model7.compile(loss=keras.losses.binary_crossentropy,
					  optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
					  metrics=['accuracy'])
		history_rms = model7.fit(X_train, y_train, 
							 validation_data=(X_test, y_test), 
							 epochs=epochs, 
							 batch_size=batch_size,
							 verbose=2)
							 
		model8 = clone_model(self.model)
		model8.compile(loss=keras.losses.binary_crossentropy,
					  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
					  metrics=['accuracy'])
		history_adam = model8.fit(X_train, y_train, 
							 validation_data=(X_test, y_test), 
							 epochs=epochs, 
							 batch_size=batch_size,
							 verbose=2)
							 
							 
		# hua tu
		
		fig = plt.figure(figsize=(12,8))
		plt.plot(range(epochs),history_default_sgd.history['val_acc'],label='Constant lr')
		plt.plot(range(epochs),history_time_sgd.history['val_acc'],label='Time-based')
		plt.plot(range(epochs),history_step_sgd.history['val_acc'],label='Step decay')
		plt.plot(range(epochs),history_exp_sgd.history['val_acc'],label='Exponential decay')
		plt.plot(range(epochs),history_agagrad.history['val_acc'],label='Adagrad')
		plt.plot(range(epochs),history_adadelta.history['val_acc'],label='Adadelta')
		plt.plot(range(epochs),history_rms.history['val_acc'],label='RMSprop')
		plt.plot(range(epochs),history_adam.history['val_acc'],label='Adam')
		plt.legend(loc=0)
		plt.xlabel('epochs')
		plt.xlim([0,epochs])
		plt.ylabel('accuracy on validation set')
		plt.grid(True)
		plt.title("Comparing Model Accuracy")
		plt.show()
		fig.savefig('img/compare-accuracy.jpg')
		plt.close(fig)
		
		# plt the loss
		fig = plt.figure(figsize=(12,8))
		plt.plot(range(epochs),history_default_sgd.history['val_loss'],label='Constant lr')
		plt.plot(range(epochs),history_time_sgd.history['val_loss'],label='Time-based')
		plt.plot(range(epochs),history_step_sgd.history['val_loss'],label='Step decay')
		plt.plot(range(epochs),history_exp_sgd.history['val_loss'],label='Exponential decay')
		plt.plot(range(epochs),history_agagrad.history['val_loss'],label='Adagrad')
		plt.plot(range(epochs),history_adadelta.history['val_loss'],label='Adadelta')
		plt.plot(range(epochs),history_rms.history['val_loss'],label='RMSprop')
		plt.plot(range(epochs),history_adam.history['val_loss'],label='Adam')
		plt.legend(loc=0)
		plt.xlabel('epochs')
		plt.xlim([0,epochs])
		plt.ylabel('loss on validation set')
		plt.grid(True)
		plt.title("Comparing Model Loss")
		plt.show()
		fig.savefig('img/compare-loss.jpg')
		plt.close(fig)