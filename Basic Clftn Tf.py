#  Task 1: Introduction
# This graph describes the problem that we are trying to solve visually.We want to create and train a model that takes an image of a hand written digit as input and predicts the class of that digit, that is, it predicts the digit or it predicts the class of the input image.
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.ERROR)
print('Using TensorFlow version', tf.__version__)

# # Task 2: The Dataset

from tensorflow.keras.datasets import mnist #mnist has datas
(x_train, y_train), (x_test, y_test) = mnist.load_data() #these are Numoy multidim arrays

# ### Shapes of Imported Arrays
print('x train shape', x_train.shape) #no. of egs,each eg has 28 rows and 28 columns
print('y train shape', y_train.shape) #28 pixels high n 28p wide #60k dim vector
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape) #10k dim vector

# ### Plot an Image Example

#TO READ FIRST TRAINING EG.
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(x_train[0], cmap='binary') #colormap just black n white
plt.show()

y_train[0]
print(set(y_train))

# # Task 3: One Hot Encoding

# After this encoding, every label will be converted to a list with 10 elements and the element at index to the corresponding class will be set to 1, rest will be set to 0:
# 
# | original label | one-hot encoded label |
# |------|------|
# | 5 | [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] |
# | 7 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] |
# | 1 | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] |

from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# ### Validated Shapes

#each eg. label was 1d now becomes a 10 dim vector
print('y_train encoded shape:', y_train_encoded.shape)
print('y_test encoded shape:', y_test_encoded.shape)

# ### Display Encoded Labels
y_train_encoded[0]

# # Task 4: Neural Networks

# ### Linear Equations
# 
# ![Single Neuron](images/1_2.png)

# y = w1 * x1 + w2 * x2 + w3 * x3 + b
# Where the `w1, w2, w3` are called the weights and `b` is an intercept term called bias. The equation can also be *vectorised* like this:
#y = W . X + b
# Where `X = [x1, x2, x3]` and `W = [w1, w2, w3].T`. The .T means *transpose*. This is because we want the dot product to give us the result we want i.e. `w1 * x1 + w2 * x2 + w3 * x3`. This gives us the vectorised version of our linear equation.

# ### Neural Networks
# ![Neural Network with 2 hidden layers](images/1_4.png)

# # Task 5: Preprocessing the Examples# 
# ### Unrolling N-dimensional Arrays to Vectors
import numpy as np
x_train_reshaped = np.reshape(x_train, (60000, 784)) #2N parameter as desires shape
x_test_reshaped = np.reshape(x_test, (10000,784))
print('x_train_reshaped:', x_train_reshaped.shape)
print('x_test_reshaped:', x_test_reshaped.shape)

# ### Display Pixel Values
print(x_train_reshaped[0])

# ### Data Normalization
x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10 #incase std is too small
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon) #note no new mean/std was calculated for testnorm, to avoid unecessary bias

# ### Display Normalized Pixel Values
print(x_train_norm[0])

# # Task 6: Creating a Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential ([
    Dense(128, activation= 'relu', input_shape=(784,)),
    Dense(128, activation= 'relu'), #no need to specify shape coz it takes above shape
    Dense(10, activation= 'softmax')
])

# ### Activation Functions# 
# The first step in the node is the linear sum of the inputs:
# Z = W . X + b
# The second step in the node is the activation function output:
# A = f(Z)
# Graphical representation of a node where the two operations are performed#
 
# ### Compiling the Model
model.compile(
    optimizer= 'sgd',
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# # Task 7: Training the Model
model.fit(x_train_norm, y_train_encoded, epochs=3)

# ### Evaluating the Model
_, accuracy = model.evaluate(x_test_norm, y_test_encoded) #evaluate gives loss, accuracy so _, ignores first returned value, here the loss
print('test set accuracy:', accuracy * 100)

# # Task 8: Predictions
# ### Predictions on Test Set
preds = model.predict(x_test_norm) #no labels coz just prdictions, we are not comparing
print('shape of predictions:', preds.shape)
#op will be 10 softmax prob pred
#highest prob score is gonna be our final class predictn using argmax

# ### Plotting the Results
#plot predictns vs actual labels of few, not 10000!
plt.figure(figsize=(12,12))
start_index = 0
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.grid(False)
    plt.xticks([]) #no x or y ticks
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i]) #forgot inside bracket preds and all pred became=0
    gt = y_test[start_index+i] #groundtruth
    
    col = 'g'
    if pred != gt:
            col = 'r'
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, gt), color=col) #col gave labels red n green else full black
    plt.imshow(x_test[start_index+i], cmap='binary') #xtestNOT NORMALIZD colr map is binary
plt.show()
plt.plot(preds[8])
plt.show()
#above are softmax prob 
