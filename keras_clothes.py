# librairies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime


print(tf.__version__) # tensorflow version (debug purpose only)

# arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--test', '-t', help='Show the template', action='store_true')
parser.add_argument('--data', '-d', help='Show a batch of data', action='store_true')

args = parser.parse_args()


fashion_mnist = tf.keras.datasets.fashion_mnist # importation of the dataset (here clothes images)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # preparation of the dataset

# prparation of clothes classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# testing the dataset
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

# show the template of an image in the dataset
if (args.test == True):
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

# define the standardization of an image 
train_images = train_images / 255.0
test_images = test_images / 255.0

# show a batch of pictures from the dataset
if (args.data == True):
    plt.figure(figsize=(10,10)) 
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# neural network creation
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # image size used
    tf.keras.layers.Dense(128, activation='relu'), # densely connected neural layers
    tf.keras.layers.Dense(10) # number of clothes classes
])



# compilation of the neural network
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# creation of the TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# training of the model, and importation of useful data in TensorBoard
model.fit(train_images, train_labels, epochs = 5, callbacks=[tensorboard_callback]) 
# an epoch is one of the data iteration

# obtention of the accuracy and the loss of the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# the accuracy is the precision of determining an object by the model, we search the highest accuracy possible
# the loss is a penalty of the model, we search for the fewest loss possible

print('\nTest accuracy:', test_acc)

# prediction of the result of the model
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]

# visual render of the data analysis
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# plot the first x test images, their predicted labels, and the true labels
# color correct predictions in blue and incorrect predictions in red
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


