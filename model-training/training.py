from tensorflow import test

from keras.applications import vgg16
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input
from keras.preprocessing.image import ImageDataGenerator


## TRAINING JOB CONFIGURATION
# Data filepaths
TRAIN_DATA_DIR = '/data/train/' # Point to directory with your data
VALID_DATA_DIR = '/data/validation/'

# Model parameters
NUM_CLASSES = 2
INPUT_SIZE = 224 # Width/height of image in pixels (224 for ResNet/VGG16, 299 for Xception model)
LEARNING_RATE = 0.0001

# For GPU training - script will check if GPU is available
BATCH_SIZE_GPU = 32 # Number of images used in each iteration
EPOCHS_GPU = 50 # Number of passes through entire dataset

# For CPU training
BATCH_SIZE_CPU = 4
EPOCHS_CPU = 1

if test.is_gpu_available(): # Check if GPU is available
    BATCH_SIZE = BATCH_SIZE_GPU # GPU
    EPOCHS = EPOCHS_GPU

else:
    BATCH_SIZE = BATCH_SIZE_CPU # CPU
    EPOCHS = EPOCHS_CPU


## CREATE DATA GENERATORS
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
        VALID_DATA_DIR,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical')


## BUILD MODEL
# Download pretrained VGG16 model and create model for transfer learning
base_model = vgg16.VGG16(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)

# Add a logistic layer
x = BatchNormalization()(x)
predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

# Model for training
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all convolutional pretrained model layers - train only top layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=["accuracy"])
model.summary()


## TRAIN MODEL
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    verbose=2)


## SAVE MODEL
model.save('model.h5')
