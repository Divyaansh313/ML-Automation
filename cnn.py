from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
import os

def model_train(epoch_no,num_of_crp):

   epochs=epoch_no
   model = Sequential()
   model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
   model.add(MaxPooling2D(pool_size=(2, 2)))

   if num_of_crp > 1:
      model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu',
                        ))
      model.add(MaxPooling2D(pool_size=(2, 2)))

   model.add(Flatten())
   model.add(Dense(units=128, activation='relu'))
   model.add(Dense(units=1, activation='sigmoid'))
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   from keras_preprocessing.image import ImageDataGenerator
   train_datagen = ImageDataGenerator(
           rescale=1./255,
           shear_range=0.2,
           zoom_range=0.2,
           horizontal_flip=True)
   test_datagen = ImageDataGenerator(rescale=1./255)
   training_set = train_datagen.flow_from_directory(
           '/dataset/cat_or_dog/training_set/',
           target_size=(64, 64),
           batch_size=32,
           class_mode='binary')
   test_set = test_datagen.flow_from_directory(
           '/dataset/cat_or_dog/test_set/',
           target_size=(64, 64),
           batch_size=32,
           class_mode='binary')
   model.fit(training_set,
           steps_per_epoch=50,
           epochs=epochs,
           validation_data=test_set,
           validation_steps=800)
   accuracy = model.evaluate(test_set,verbose=0)[1]
   accuracy = accuracy*100
   return accuracy

no_of_epoch = 1
no_of_layer = 1
my_model = model_train(no_of_epoch,no_of_layer)

f = open("accuracy.txt","w+")
f.write(str(my_model))
f.close()
os.system("mv /accuracy.txt /dataset/")
