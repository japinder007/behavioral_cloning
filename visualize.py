import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

""" Visualize the histogram of steering values. 

The training set most likely has a pre-ponderence of values closer to zero and 
that makes the model learn poorly. 

:param train_augmentations: A list of AugmentedImage objects. 
:rtype: void 
"""
def plot_distribution(train_augmentations, file_name):
	steering_values = [a.steering_value() for a in train_augmentations]
	plt.figure()
	plt.hist(steering_values, bins=100, alpha=0.75, facecolor='g', range=(-1,1))
	plt.grid(True)
	plt.title('Distribution of Steering Values')
	plt.savefig(file_name)

"""
Given a model and sample training example, plot the abs eror wrt original y values.

:param model - The model to evaluate.
:param X_sample - features.
:param y_sample - expected output.
:rtype: void
"""
def plot_mse_sample(model, X_sample, y_sample, file_name='sample_scatter.png'):
	predictions = model.predict(X_sample)
	print('Shape of features: {f}, shape of y_sample {y}, shape of predictions {p}'.format(f=X_sample.shape, p=predictions.shape, y=y_sample.shape))
	sample_error = predictions - np.expand_dims(y_sample, axis=1)
	print('predictions: {p}, error : {e}'.format(p=predictions.shape, e=sample_error.shape))
	plt.figure()
	plt.title('Mse vs. steering values')
	plt.scatter(y_sample, sample_error.reshape(y_sample.shape[0]))
	plt.xlabel("Steering Values")
	plt.ylabel("MSE")
	plt.grid(True)
	plt.savefig(file_name)