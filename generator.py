from augmentation import create_augmentations
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_test(augmentations, test_size=0.2, plot_train_histogram=True, histogram_file='train_augmentations.png'):
		train_augmentations, validation_augmentations = train_test_split(augmentations, test_size=test_size)
		print('')
		print('Training data size {t}'.format(t=len(train_augmentations)))	
		print('Validation data size {v}'.format(v=len(validation_augmentations)))
		return (train_augmentations, validation_augmentations)

def get_XY_from_augmentations(augmentations):
	images = []
	steering_values = []
	for a in augmentations:			
		image = a.image()
		images.append(image)
		steering_values.append(a.steering_value())

	X_train = np.array(images)
	y_train = np.array(steering_values)
	return X_train, y_train

def generator(augmentations, batch_size=128):
	num_samples = len(augmentations)
	while 1:
		current_augmentations = shuffle(augmentations)
		for offset in range(0, num_samples, batch_size):
			augmentation_slice = augmentations[offset:offset + batch_size]
			X_train, y_train = get_XY_from_augmentations(augmentation_slice)
			yield shuffle(X_train, y_train)	

def generate_XY_samples(augmentations, sample_fraction=0.1):
	num_samples = int(sample_fraction * len(augmentations))
	samples = shuffle(augmentations)[: num_samples]
	return get_XY_from_augmentations(samples)
	
