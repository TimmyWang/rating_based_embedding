import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dot
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Activation,Softmax,Multiply
from keras.activations import tanh, relu, sigmoid
#from keras.initializers import RandomUniform
from keras import backend as K






def get_model(num_user, num_item, latent_dim):

	user_index = Input(shape=(1,))
	item_index = Input(shape=(1,))

	user_embedding = Embedding(num_user, latent_dim, name="user_embedding")(user_index)
	item_embedding = Embedding(num_item, latent_dim, name="item_embedding")(item_index)
	user_embedding = Reshape((latent_dim,))(user_embedding)
	item_embedding = Reshape((latent_dim,))(item_embedding)
	#user_embedding = Dropout(0.2)(user_embedding)
	#item_embedding = Dropout(0.2)(item_embedding)

	# Approach 1
	#concat_embedding = Concatenate(axis=-1)([user_embedding,item_embedding])
	#dense = Dense(latent_dim,activation="relu")(concat_embedding)
	#output = Dense(1,activation="linear")(dense)

	# Approach 2
	output = Dot(axes=-1)([user_embedding,item_embedding])
	output = Activation(tanh)(output)
	output = Lambda(lambda x: (x+1)*2+1)(output)
	
	# Approach 3
	#output = Dot(axes=-1)([user_embedding,item_embedding])
	#output = Activation(sigmoid)(output)
	#output = Lambda(lambda x: x*4+1)(output)

	return Model([user_index,item_index],output)





