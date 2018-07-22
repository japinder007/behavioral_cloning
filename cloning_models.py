from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dense, MaxPooling2D
from keras.layers import Cropping2D
from generator import generator
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam

def normalize_and_crop(model, input_shape):
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	return model

# Splits data (augmentations) into training and test set. 
# Returns a (model, history_object) tuple.
def add_fit_generator(model, train_augmentations, validation_augmentations, epochs, batch_size=32, test_size=0.2):
	print('')
	print('batch_size ', batch_size)
	print('steps_per_epoch : ', len(train_augmentations) // batch_size)
	print('steps_per_epoch (val) : ', len(validation_augmentations) // batch_size)
	print('')
	train_generator = generator(train_augmentations, batch_size)
	validation_generator = generator(validation_augmentations, batch_size)
	history_object = model.fit_generator(
		train_generator, 
		steps_per_epoch=len(train_augmentations) // batch_size, 
		validation_data=validation_generator, 
		validation_steps=len(validation_augmentations) // batch_size,
		epochs=epochs,
		verbose=2
	)
	return (model, history_object)


def basic_model(augmentations, input_shape=(160, 320, 3)):
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Flatten())
	model.add(Dense(1))
	return model

def lenet(augmentations, input_shape=(160, 320, 3)):
	model = Sequential()

	model = normalize_and_crop(model, input_shape)	
	model.add(Conv2D(6,(5,5),activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(16,(5,5),activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	return model

def nvidia(augmentations, input_shape=(160, 320, 3)):
	model = Sequential()
	
	model = normalize_and_crop(model, input_shape)
	model.add(Conv2D(24,(5,5),strides=(2, 2), activation='relu'))
	model.add(Conv2D(36,(5,5),strides=(2, 2), activation='relu'))
	model.add(Conv2D(48,(5,5),strides=(2, 2), activation='relu'))
	model.add(Conv2D(64,(3,3),activation='relu'))
	model.add(Conv2D(64,(3,3),activation='relu'))
	
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	return model

def elu(augmentations, input_shape=(160, 320, 3)):
	model = Sequential()

	# Normalize
	model = normalize_and_crop(model, input_shape)

	# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
	model.add(Conv2D(24, (5, 5), subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(36, (5, 5), subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(48, (5, 5), subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())

	#model.add(Dropout(0.50))

	# Add two 3x3 convolution layers (output depth 64, and 64)
	model.add(Conv2D(64, (3, 3), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(64, (3, 3), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())

	# Add a flatten layer
	model.add(Flatten())

	# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
	model.add(Dense(100, W_regularizer=l2(0.001)))
	model.add(ELU())
	#model.add(Dropout(0.50))
	model.add(Dense(50, W_regularizer=l2(0.001)))
	model.add(ELU())
	#model.add(Dropout(0.50))
	model.add(Dense(10, W_regularizer=l2(0.001)))
	model.add(ELU())
	#model.add(Dropout(0.50))

	# Add a fully connected output layer
	model.add(Dense(1))
	return model


