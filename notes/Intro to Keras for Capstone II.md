
These are my notes from the documentations on the keras.io website. Specifically the introduction to Keras for engineers in the getting started section.

## What is Keras

Keras in an API designed for human beings. It follows best practices for reducing cognitive load by offering consistent and simple APIs. 

It is built on top of the TensorFlow platform. Can be deployed anywhere by exporting models to JavaScript to run directly in browser or to TF Life to run on mobile or embedded devices. It is also easy to serve Keras models via a web API.

## Data loading & preprocessing

Neural networks process vectorized & standardized representations of data. For image files that is relevant to our project, they need to be read and decoded into integer tensors, then converted to floating point and normalized to small values (between 0 and 1).

Data also needs to be in one of the following formats for Keras models to accept them as input:
1. **NumPy arrays**, good option if data fits in memory
2. **TensorFlow `Dataset` objects**, high-performance option that is more suitable for datasets that do not fit in memory and rather streamed from disk or filesystem.
3. **Python generators** that yield batches of data.

### Data loading

Suppose we have image files sorted by class in different folders, we can do this to load the data
```python
# Create a dataset
dataset = keras.utils.image_dataset_from_directory(
	'path/to/main_directory', batch_size=64, image_size=(200,200))

# Iterate over the batches yielded by the dataset.
for data, labels in dataset:
	print(data.shape)   # (64, 200, 200, 3)
	print(data.dtype)   # float 32
	print(labels.shape) # (64,)
	print(labels.dtype) # int32
```

The label of a sample by default will be the rank of its folder in alphanumeric order. They can also be passed by using `class_names=['class_a', 'class_b']`

## Data preprocessing

**The ideal machine learning model is end-to-end**, this means that data preprocessing should be part of the model we export. 

The ideal model should expect as input something as close as possible to raw data, in our case the RGB pixel values in the $[0,225]$ range of the microscope's images.

### Keras preprocessing layers

These layers can be included directly into the model, making it portable. The two layers we most likely want are feature normalization via `Normalization` layer and image rescaling, cropping, and image data augmentation.

Below are two examples of normalizing features and image rescaling and center-cropping

```python
from tensorflow.keras.layers import Normalization

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))   # var: 1.0005
print("mean: %.4f" % np.mean(normalized_data)) # mean: 0.0000
```

```python
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

# Example image data, with values in the [0, 255] range
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
```

## Building models with the Keras Functional API

Layers are simple input-output transformations. A "model" is a directed acyclic graph of layers. 

The most common and most powerful way to build Keras models is the Functional API. You start by specifying the shape of your inputs, in our case the input for a $200\times200$ RGB image would have shape `(200, 200, 3)`.

```python
inputs = keras.Input(shape=(200, 200, 3))
```

Example of chaining layer transformations on top of the inputs until we get a final output.

```python
from tensorflow.keras import layers

# Center-crop images to 150x150
x = CenterCrop(height=150, width=150)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(x)

# Apply some convolution and pooling layers
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

then we can instantiate a `Model` object. The model behaves as a bigger layer an we can call it on batches of data.

```python
model = keras.Model(inputs=inputs, outputs=outputs)

data = np.random.randint(0, 256, size=(64,200,200,3)).astype("float32")
processed_data = model(data)

model.summary() # prints a summary of how data gets transformed each stage
```

## Training models with `fit()`

The `Model` class has a built-in training loop called `fit()`. 

Before calling `fit()` we need to specify an optimizer and loss function. (review this separately). Loss and optimizer functions can be specified using their string identifiers. Once the model is compiled, we can start fitting the model to the data.

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# with numpy data
model.fit(numpy_array_of_samples, numpy_array_of_labels,
		 batch_size=32, epochs=10)
		 
# with a Dataset
model.fit(dataset_of_samples_and_labels, epochs=10)
```

## Finding the best model configuration with hyperparameter tuning

Leveraging a systematic approach to optimizing the configuration (architecture choices, layer sizes, etc.) using a hyperparameter search.


# End-to-end Image Classification Example

https://keras.io/examples/vision/image_classification_from_scratch/

https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py

## Using image data augmentation

Since we will not have a large image dataset, introducing random and realistic transformations to our training images will artificially increase sample diversity. Such as horizontal flipping or random rotations. Below is an example of applying random horizontal flips to images.

```python
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
```

