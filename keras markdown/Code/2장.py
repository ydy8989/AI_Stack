from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist

(train_img, train_lab), (test_img, test_lab) = mnist.load_data()


network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
network.add(layers.Dense(10, activation = 'softmax'))

network.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

train_img = train_img.reshape((60000, 28 * 28))
train_img = train_img.astype('float32')/255

test_img = test_img.reshape((10000, 28 * 28))
test_img = test_img.astype('float32')/255

train_lab = to_categorical(train_lab)
test_lab = to_categorical(test_lab)

network.fit(train_img, train_lab, epochs = 5, batch_size = 128)

test_loss, test_acc = network.evaluate(test_img, test_lab)
print('test_acc:', test_acc)

