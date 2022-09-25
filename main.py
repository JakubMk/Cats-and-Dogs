# This is a sample Python script.
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pre_trained_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights='imagenet'
    )

    for layer in pre_trained_model.layers:
        layer.trainable = False

    pre_trained_model.summary()

    # Choose 'mixed7' as a last layer of your base model
    last_layer = pre_trained_model.get_layer('mixed7')
    print('Last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    # Add dense layer for the classifier
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a FC layer with 1024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    # Append the dense network to the base model
    model = Model(pre_trained_model.input, x)

    # Print the model summary
    model.summary()

    # Set the training parameters
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # PREPARE THE DATASET
    base_dir = 'cats_and_dogs_filtered'

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Directory with training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')

    # Directory with training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')

    # Directory with validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')

    # Directory with validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(150, 150))

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100,
        epochs=20,
        validation_steps=50,
        verbose=2
    )

    # Print train and validation accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
