import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import keras
# Load the events.csv in a pandas DataFrame:
events = pd.read_csv('events.csv')

# Pre-processing the data
# We convert view events to 1, addtocart events to 2, and transaction events to 3 with the following code:
events.event.replace(to_replace=dict(view=1,
                                     addtocart=2,
                                     transaction=3),
                     inplace=True)

# Drop the transcationid and timestamp columns that we don't need:
events.drop(['transactionid'],axis=1,inplace=True)
events.drop(['timestamp'],axis=1,inplace=True)

# Shuffle the dataset to get random data for training and test datasets:
# events = events.reindex(np.random.permutation(events.index))
events = events.sample(frac=1).reset_index(drop=True)

# Split the data in train, valid, and test sets, as follows:
split_1 = int(0.8 * len(events))
split_2 = int(0.9 * len(events))
train = events[:split_1]
valid = events[split_1:split_2]
test = events[split_2:]
print(train.head())
print(valid.head())
print(test.head())


# Now let's create a matrix factorization model in Keras:
# Store the number of visitors and items in a variable, as follows:
n_visitors = events.visitorid.nunique()
n_items = events.itemid.nunique()

# Set the number of latent factors for embedding to 5. You may want to try different values to see the impact on the model training:
n_latent_factors = 5

# Import the Input, Embedding, and Flatten layers from the Keras library:
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense

# Start with the items – create an input layer for them as follows:
item_input = Input(shape=[1],name='Items')

# Create an Embedding representation layer and then flatten the Embedding layer to get the output in the number of latent dimensions that we set earlier:
item_embed = Embedding(n_items +1,
                       n_latent_factors,
                       name='ItemsEmbedding')(item_input)
item_vec = Flatten(name='ItemsFlatten')(item_embed)

# Similarly, create the vector space representation for the visitors:
visitor_input = Input(shape=[1],name='Visitors')
visitor_embed = Embedding(n_visitors +1,
                          n_latent_factors,
                          name='VisitorsEmbedding')(visitor_input)
visitor_vec = Flatten(name='VisitorsFlatten')(visitor_embed)

# Create a layer for the dot product of both vector space representations:
dot_prod = keras.layers.dot([item_vec, visitor_vec],axes=[1,1],
                            name='DotProduct')

# Build the Keras model from the input layers, and the dot product layer as the output layer, and compile it as follows:
model = keras.Model([item_input, visitor_input], dot_prod)
model.compile('adam', 'mse')
model.summary()

# Since the model is complicated, we can also draw it graphically using the following commands:
keras.utils.plot_model(model,
                       to_file='matrix factorization.png',
                       show_shapes=True,
                       show_layer_names=True)
from IPython import display
display.display(display.Image('matrix factorization.png'))


# Now let's train and evaluate the model: 
model.fit([train.itemid,train.visitorid], train.event , epochs=50)
score = model.evaluate([ test.itemid,test.visitorid], test.event)
print('mean squared error:', score)
# mean squared error: 0.9643871370311828

# Now, let's build the neural network model to provide the same recommendations
n_lf_visitor = 5
n_lf_item = 5

# Build the item and visitor embeddings and vector space representations the same way we built earlier:
item_input = Input(shape=[1],name='Items')
item_embed = Embedding(n_items + 1,
                       n_lf_item,
                       name='ItemsEmbedding')(item_input)
item_vec = Flatten(name='ItemsFlatten')(item_embed)
visitor_input = Input(shape=[1],name='Visitors')
visitor_embed = Embedding(n_visitors + 1,
                          n_lf_visitor,
name='VisitorsEmbedding')(visitor_input)
visitor_vec = Flatten(name='VisitorsFlatten')(visitor_embed)

# Instead of creating a dot product layer, we concatenate the user and visitor representations, and then apply fully connected layers to get the recommendation output:
concat = keras.layers.concatenate([item_vec, visitor_vec],
                                  name='Concat')
fc_1 = Dense(80,name='FC-1')(concat)
fc_2 = Dense(40,name='FC-2')(fc_1)
fc_3 = Dense(20,name='FC-3', activation='relu')(fc_2)
output = Dense(1, activation='relu',name='Output')(fc_3)

#Define and compile the model as follows:
optimizer = keras.optimizers.Adam(lr=0.001)
model = keras.Model([item_input, visitor_input], output)
model.compile(optimizer=optimizer,loss= 'mse')

#Train and evaluate the model:
model.fit([train.visitorid, train.itemid], train.event, epochs=50)
score = model.evaluate([test.visitorid, test.itemid], test.event)
print('mean squared error:', score)
# mean squared error: 0.057942360164440876

model.summary()

# Since the model is complicated, we can also draw it graphically using the following commands:
keras.utils.plot_model(model,
                       to_file='neural network.png',
                       show_shapes=True,
                       show_layer_names=True)
from IPython import display
display.display(display.Image('neural network.png'))

# Save the model
model.save("model")

