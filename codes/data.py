import numpy as np 





class FakeData:

	def __init__(self, dimensions):		
		self.dimensions = dimensions
		self.meta_users = []
		self.meta_items = []
		self.users = []
		self.items = []
		self.ratings = None
		self.rating_percentile = None
		self.train_ratio = None
		self.train = [[],[],[]] # user_index, item_index, rating
		self.test = [[],[],[]]

	def add_user(self, preference, num, stdev):		
		assert len(preference) == self.dimensions
		self.meta_users.append((preference, num, stdev))

	def add_item(self, feature, num, stdev):		
		assert len(feature) == self.dimensions
		self.meta_items.append((feature, num, stdev))

	def generate_data(self, rating_percentile=(10,30,70,90), train=0.05):
		self.rating_percentile = rating_percentile
		self.train_ratio = train
		self._generate_embedding()
		self._generate_rating()
		self._generate_train_test_data()

	# for ploting purpose later
	def get_color_list(self,color_list,user=True):
		if user:
			num_list = list(zip(*self.meta_users))[1]
		else:
			num_list = list(zip(*self.meta_items))[1]
		assert len(color_list) == len(num_list)
		return [c for i,c in enumerate(color_list) for _ in range(num_list[i])]

	def _generate_embedding(self):
		
		for preference, num, stdev in self.meta_users:
			for _ in range(num):
				self.users.append([np.random.normal(loc=p, scale=stdev) for p in preference])

		for feature, num, stdev in self.meta_items:
			for _ in range(num):
				self.items.append([np.random.normal(loc=f, scale=stdev) for f in feature])

		self.users = np.array(self.users)
		self.items = np.array(self.items)

	def _generate_rating(self):
		
		rating = np.matmul(self.users, self.items.transpose())
		shape = rating.shape
		rating_flat = rating.reshape(1,shape[0]*shape[1])
		boundaries = [np.percentile(rating_flat, pt) for pt in self.rating_percentile]
		
		def normailze(n):
			for i, b in enumerate(boundaries):
				if n < b:
					return i + 1
			return len(self.rating_percentile) + 1

		normalize = np.vectorize(normailze)

		self.ratings = normalize(rating)

	def _generate_train_test_data(self):

		for user in range(len(self.users)):
			for item in range(len(self.items)):
				if np.random.uniform(0,1) <= self.train_ratio:
					self.train[0].append(user)
					self.train[1].append(item)
					self.train[2].append(self.ratings[user,item])
				else:
					self.test[0].append(user)
					self.test[1].append(item)
					self.test[2].append(self.ratings[user,item])

		self.train = np.array(self.train)
		self.test = np.array(self.test)








