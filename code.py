import os 
import cv2 
import numpy as np
import glob as gb
import matplotlib.pyplot as plt
import tensorflow as tf

Datasetpath = 'Data/'

code = {'0':0 ,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
def getcode(n):
    for x , y in code.items():
        if n == y:
            return x
        
X_train = []
y_train = []
for folder in os.listdir(Datasetpath + 'digits_train_set'):
    files = gb.glob(pathname = Datasetpath + 'digits_train_set//' + folder + '/*jpg')
    for file in files:
        image = cv2.imread(file)
        #image_array = np.resize(28, 28)
        X_train.append(list(image))
        y_train.append(code[folder])       
        
X_test = []
y_test = []
for folder in os.listdir(Datasetpath + 'digits_test_set'):
    files = gb.glob(pathname = Datasetpath + 'digits_test_set//' + folder + '/*jpg')
    for file in files:
        image = cv2.imread(file)
        #image_array = np.resize(28, 28)
        X_test.append(list(image))
        y_test.append(code[folder])
        
        
X_pred = []
files = gb.glob(pathname= str(Datasetpath + 'digits_pred_set/*.jpg'))
for file in files:
    image = cv2.imread(file)
    X_pred.append(list(image))
        

        
X_train = np.array(X_train)
X_test = np.array(X_test)
X_pred_array = np.array(X_pred)

y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')

'''
model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(28,28,3)))
model.add(tf.keras.layers.Conv2D(110,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dense(10, activation ='softmax'))

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics=['accuracy'])

model.fit(X_train, y_train , epochs = 3)

model.save('handwrtten.model')
'''
model = tf.keras.models.load_model('handwritten.model')



loss ,accuracy = model.evaluate(X_test,y_test)

print(f'test loss value is {loss}')
print(f'Accuracey Value is {accuracy}')



y_pred = model.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))

y_result = model.predict(X_pred_array)

print('Prediction Shape is {}'.format(y_result.shape))


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))






