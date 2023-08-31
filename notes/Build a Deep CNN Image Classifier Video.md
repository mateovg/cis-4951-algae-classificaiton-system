
Notes from this [video](https://www.youtube.com/watch?v=jztwpsIzEGc). The code is from the video and the rest of the text is my own explanation and relation it to our algae classification project.

## Setup

### Dependencies and Setup

Learn how to configure a virtual environment for this. Maybe use PyCharm?

```
!pip install tensorflow tensorflow-gpu opencv-python matplotliib 
```

```python
import tensorflow as tf
import os
```

```python
# avoid OOM errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
```

Keeps memory to the absolute minimum needed.

### Clean data

We need to see what our actual images look like. Since we're creating the training data ourselves, it shouldn't be a problem. We should however use the data augmentation techniques described in the Keras guide to increase the size of our training set. Meaning rotate/flip training images.

`cv2` stores images as BGR not RBG, might be needed at some point.

### Load data

Using the TensorFlow `Dataset` object. Allows us to use data pipelines to scale much better and "makes stuff a lot cleaner". We will use a Keras utility that will use it, rather than accessing the API directly.

```python
data = tf.keras.utils.image_dataset_from_directory('data')
```

Will build an image dataset on the fly, no need to make labels and will also perform some preprocessing out of the box.

Some default values are: `batch_size=32, image_size(256,256), suffle=True`

```python
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# batch[0] = images
# batch[1] = labels
```

Allows us to access the generator from the pipeline. Using `.next()` to get consecutive batches. Might not be necessary for our current model size but good practice.

```python
# example to display 4 images
# their label as the title of the class
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
	ax[idx].imshow(img)
	ax[idx].title.set_title(batch[1][idx])
```

## Preprocess Data

### Scale Data

```python
data = data.map(lambda x,y: (x/255, y))
```

Using `map` allows us to perform the operation in pipeline. The lambda function is normalizing all the RGB values to be between $0$ and $1$. 

`tf.data.DataSet` has a ton of functions to perform transformations, we want to use theses as part of the pipeline. 

### Split Data

Split into testing and training partitions. These are the number of batches to use for each, so make sure the numbers are not $0$.

```python
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

# use take and skip methods in DataSet pipeline
# data is already shuffled, if not, shuffle it before this
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
```

## Build Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
```

`Sequential` is good for us since we have one input and one output and much simpler than the functional (?) model.

* `Conv2D` is a 2D convolutional layer, spatial convolution over images.
* `MaxPooling2D` acts like a condensing layer and returns the max value from regions from the convolution.
* `Flatten`, reduces the channels from the image data into a single layer that `Dense` can take to output a single output, the classification.

This forms the architecture of the NN.

```python
# instantiate sequential model object
model = Sequential()

# add in layers sequentially
# first layer is an input layer, 16 3x3 filters with a stride of 1
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape = (256, 256, 3)))
model.add(MaxPooling2D()) # max value of (2,2) regions, condenses images

# 32 3x3 filters with a stride of 1
model.add(Conv2D(32, (3,3), 1, activation='relu')) # same as last block
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu')) 
model.add(MaxPooling2D())

model.add(Flatten()) # condeses the dimensions to a single dimension 

model.add(Dense(256, activation='relu')) # fully connected layers (Dense) 
model.add(Dense(1, activation='sigmoid'))
```

Hyperparameters can be tuned but these are good to start out with. Convolution needs the number of filters, the size, and the stride. As well as the activation function. `relu` allows us to account for non-linear patterns which is what makes the NN so powerful. 

`sigmoid` maps values to $0$ or $1$, we don't want this since we have more than just a binary classification. Need to research how to use this for classifying multiple labels.

The convolution layers could use padding to preserve the shape of the images.

```python
# adam optimizer
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
```

Again, we are not doing a binary classification so this will have to be different.

```python
model.summary()
```

### Train Model

```python
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorbord_callbacks])
```

Saving the call backs in the `hist` variable will let us plot the performance over the training time which will be super cool for the poster.

### Plot Performance

```python
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legen(loc='upper left')
plt.show()
```

## Evaluate Performance