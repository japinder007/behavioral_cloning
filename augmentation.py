from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dense, MaxPooling2D
import csv
import cv2
import os

def read_image(image_dir, row, index):
	file = row[index].split('/')[-1]
	path = os.path.join(image_dir, file)
	return cv2.imread(path)

# Main augmentation method for images. Does two types of augmentations:
# a) Flip an image horizontally.
# b) Use the left and right camera images and correct them a bit as if they were
#    the center image. other_camera_displacement is the correction.
CORRECTIONS_MULTIPLIER = [0, 1, -1]
def augment_images(csv_file_name, image_dir, other_camera_displacement=0.2):
	images = []
	steering_values = []
	with open(csv_file_name, 'r') as csv_file:
		driving_log = csv.reader(csv_file)
		index = 0
		for row in driving_log:
			index += 1
			if index == 1:
				continue
			if index % 100 == 0:
				print('Completed {r} rows'.format(r=index))

			curr_images = [read_image(image_dir, row, i) for i in range(3)]
			steering_value = float(row[3])
			curr_steering_values = [(steering_value + c * other_camera_displacement) for c in CORRECTIONS_MULTIPLIER]

			# Add the flipped images.
			curr_images_flipped = [cv2.flip(image, 1) for image in curr_images]
			curr_steering_values_flipped = [-v for v in curr_steering_values]

			images.extend(curr_images)
			steering_values.extend(curr_steering_values)

			images.extend(curr_images_flipped)
			steering_values.extend(curr_steering_values_flipped)

		print('Original images {o}, augmentations {a}'.format(o=index, a=len(images)))
	return (images, steering_values)

class AugmentedImage:
	def __init__(self, image_dir, file_name, steering_value, needs_flip = False):
		self._image_dir = image_dir
		self._file_name = file_name
		self._steering_value = steering_value
		self._needs_flip = needs_flip

	def image(self):
		path = os.path.join(self._image_dir, self._file_name)
		image = cv2.imread(path)
		return cv2.flip(image, 1) if self._needs_flip else image

	def steering_value(self):
		return self._steering_value

	def __str__(self):
		return 'f: {f}, i: {i}, s: {s}, n: {n}'.format(
			f=self._file_name, i = self._image_dir,
			s=self._steering_value, n = self._needs_flip
		)

# Reads 'csv_file_name' and for each row:
#	- Adds center, left and right images. The steering values for adjustments are adjusted for left
#     and right images.
#   - Adds flipped versions of the above three images.
def create_augmentations(csv_file_name, image_dir, other_camera_displacement=0.2):
	augmentations = []
	with open(csv_file_name, 'r') as csv_file:
		driving_log = csv.reader(csv_file)
		index = 0
		for row in driving_log:
			index += 1
			if index == 1:
				continue
			steering_value = float(row[3])
			curr_steering_values = [(steering_value + c * other_camera_displacement) for c in CORRECTIONS_MULTIPLIER]
			files = [row[i].split('/')[-1] for i in range(3)]			
			curr_augmentations = [AugmentedImage(image_dir, files[i], curr_steering_values[i]) for i in range(3)]

			# Add the flipped images.
			curr_augmentations_flipped = [
				AugmentedImage(image_dir, files[i], -1.0 * curr_steering_values[i], True) for i in range(3)
			]
			augmentations.extend(curr_augmentations + curr_augmentations_flipped)

		print('Original images {o}, augmentations {a}'.format(o=index, a=len(augmentations)))
	
	return augmentations