import numpy as np

class Sampler:
	"""Tries to provide a training set balanced across buckets.

	For training set where the predicted variable is a continuous variable,
	this class returns a more balanced trainig set. The training set if first
	split into n_buckets and the training set is balanced by down sampling
	buckets which are more frequent than n_mean = len(X_train) / n_buckets.
	These classes are down sampled s.t. they have ~ n_mean elements. Buckets
	which have frequency less than n_mean are not changed. Finally the buckets
	are merged to produce the final training set.

	"""

	def __init__(self, y_train, n_buckets):
		""" Constructor for the sampler class.
		Args:
			y_train (numpy.darray[float])  - Array of steering values.
			n_buckets (int)      		   - Number of buckets into which the range of y_train is split.
		"""
		self.y_train = y_train
		self.n_buckets = n_buckets
		self.y_min = np.min(self.y_train)
		self.y_max = np.max(self.y_train)
		self.bucket_width = (self.y_max - self.y_min) / self.n_buckets
		# Store the indices of elements satisfying the conditions for buckets.
		self.bucket_count = [0 for i in range(self.n_buckets)]
		for i in range(y_train.shape[0]):
			bucket = self.__value_to_bucket(y_train[i])
			self.bucket_count[bucket] += 1
		self.average_count = y_train.shape[0] // self.n_buckets
		self.sampling_probability_ = [
			1.0 if (self.bucket_count[i] < self.average_count) else self.average_count / self.bucket_count[i] for i in range(self.n_buckets)
		]

	def __value_to_bucket(self, value):
		b = int((value - self.y_min) / self.bucket_width)
		return min(b, self.n_buckets - 1)

	def sampling_probability(self, value):
		bucket = self.__value_to_bucket(value)
		return self.sampling_probability_[bucket]


if __name__ == '__main__':
	y_train = np.array([0.0, 0.1, 0.15, 0.18, 0.2, 0.25, 0.4])
	sampler = Sampler(y_train, 2)
	print([sampler.sampling_probability(v) for v in y_train])
