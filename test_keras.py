import tensorflow as tf

print("version= ", tf.__version__)

# chargement du jeu de donné MNIST
mnist = tf.keras.datasets.mnist

# préparation du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# création du modèle Keras par empilement de couches
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# renvoie de vecteurs <scores logits> ou <log-odds>
predictions = model(x_train[:1]).numpy()
predictions

# conversion des logits en probabilités
tf.nn.softmax(predictions).numpy()

# fonction definissant les pertes scalaires
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# perte nulle si le modèle est sûr
loss_fn(y_train[:1], predictions).numpy()

# compilation du modèle
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])