# This is a file which trains a very basic model to test the
# end to end flow for loading a model into the simulator.

import argparse
import csv
import cv2
import numpy as np
from pprint import pprint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
import os
import sys

from cloning_models import lenet, basic_model, nvidia, elu, add_fit_generator
from augmentation import create_augmentations
from generator import split_train_test, generate_XY_samples
from visualize import plot_distribution, plot_mse_sample
from sampler import Sampler

parser = argparse.ArgumentParser(description='Model Trainer.')
parser.add_argument(
	'-c', '--csv_file', type=str, help='csv file containing training data', 
	default='data/driving_log.csv'
)
parser.add_argument(
	'-i', '--image_dir', type=str, help='Directory containing images',
	default='data/IMG'
)
parser.add_argument(
	'-m', '--model', help='Name of the model (basic|lenet|nvidia|elu)', default='nvidia'
)
parser.add_argument(
	'-s', '--suffix', help='Suffix to add to the model name file', default=''
)
parser.add_argument(
	'-e', '--epochs', help='Number of epochs to optimize the model for', default=3, type=int
)
parser.add_argument(
	'--visualize_only', help='If true, exit after visualizing data', action='store_const', const=True
)
parser.add_argument(
	'-f', '--file_histogram', help='File to store the distribution of steering values', default='train_steering.png', type=str
)
parser.add_argument(
	'-b', '--buckets', help='Number of buckets', default=25, type=int
)
parser.add_argument(
	'-o', '--output_model_file', help='File to output the model in', default='model.h5', type=str
)

def should_sample(sampling_probability):
	return np.random.uniform() <= sampling_probability

def main():
	args = parser.parse_args()

	print('')
	print('-------------------------------------------------')
	print('Reading csv file {c}'.format(c=args.csv_file))
	print('Image directory {d}'.format(d=args.image_dir))
	print('Using model {m}'.format(m=args.model))
	if args.output_model_file:
		output_file = args.output_model_file
	else:
		output_file = args.model + '{s}_{e}.h5'.format(s=args.suffix, e=args.epochs)
	print('Will write model to {o}'.format(o=output_file))
	print('Will run model for {e} epochs'.format(e=args.epochs))
	print('-------------------------------------------------')
	print('')

	# Map from model name to the function which implements the model.
	name_to_model_function_map = {
		'basic' : basic_model,
		'lenet' : lenet,
		'nvidia': nvidia,
		'elu': elu
	}

	if args.model not in name_to_model_function_map:
		print('Unsupported model {m}. Only support {s}'.format(
			m=args.model, s=','.join(name_to_model_function_map.keys()))
		)
		return

	# Take each image and also generate horizontally flipped images. Do this for the left and right
	# images in addition to the center image.
	augmentations = create_augmentations(args.csv_file, args.image_dir, other_camera_displacement=0.2)
	print('Training data has {n} images '.format(n=len(augmentations)))

	# Sample the augmentations to get a balanced distribution of the images.
	# In the original training set, the majority of images are captured when
	# the car is driving straight. I have broken up the steering ranges into
	# buckets and have downsampled the buckets which have way more samples than
	# the rest. The imbalance was causing the model to give more importance to
	# the frequent buckets and not learn much from others.
	steering_values = np.array([a.steering_value() for a in augmentations])	
	sampler = Sampler(steering_values, args.buckets)
	sampled_augmentations = list(filter(lambda a: should_sample(sampler.sampling_probability(a.steering_value())), augmentations))
	print('After sampling training data has {n} images '.format(n=len(sampled_augmentations)))
	
	model = name_to_model_function_map[args.model](sampled_augmentations)
	model.compile(loss='mse', optimizer='adam')
	training_augmentations, validation_augmentations = split_train_test(sampled_augmentations)
	
	print('Plotting histogram of training steering values ...')
	plot_distribution(training_augmentations, args.file_histogram)
	print('Done plotting histogram')
	print('')
	if args.visualize_only:
		print('Exiting early after visualization')
		sys.exit(0)

	# Add a generator to load images in batches.
	model, history_object = add_fit_generator(model, training_augmentations, validation_augmentations, args.epochs)
	model.save(output_file)
	print('Model written file {m}'.format(m=output_file))
	print(history_object.history)
	
	# Print generating predictions over the training set.
	print('Plotting MSE vs steering values')
	X_sample, y_sample = generate_XY_samples(augmentations, 0.1)
	plot_mse_sample(
		model, X_sample, y_sample, file_name='sample_scatter_{e}_{s}'.format(e=args.epochs, s=args.suffix)
	)
	print('Done plotting MSE vs steering values')

if __name__ == '__main__':
	main()

