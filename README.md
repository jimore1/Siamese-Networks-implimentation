# Siamese Networks implimentation

* import the models
```python
    import re
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.layers import Activation
    from tensorflow.python.keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
    from tensorflow.python.keras.models import Sequential, Model
    from tensorflow.python.keras.optimizers import rmsprop_v2 as RMSprop
    import matplotlib.pyplot as plt
    from tensorflow.python.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D
    from keras.layers import merge
    from keras.utils.np_utils import to_categorical
    from tensorflow.python.keras.models import Model, Sequential
    from tensorflow.python.keras.regularizers import l2
    from tensorflow.python.keras.optimizers import adam_v2 as Adam
    from tensorflow.python.keras.losses import binary_crossentropy
    import numpy.random as rng
    import numpy as np
    import os
    import dill as pickle
    import matplotlib.pyplot as plt
    from sklearn.utils import shuffle
    from tensorflow.python.keras.optimizer_v2 import adam
```
* define the labels of all the categories
```python
    classes=["have","none"]
```
* define a function called `read_image` to read the image data and normalize their sizes to the same style
```python
    def read_image(filename):
        f=Image.open(filename)
        f=f.resize((100,100))
        f=np.array(f)
        f=f.reshape(3,100,100)
        return np.array(f)
```
* test the function `read_image`
```python
    img = read_image('C:\\Users\\jimore\\Downloads\\dataset\\have\\1.jpg')
    plt.imshow(img.reshape(100,100,3))
    plt.show()
```
![](data:image/png;base64,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0tJS9fX1aeXKlZo/f74k0U8e8F0AQmbNzc3avXu3tm7darspvtPZ2alFixZp8+bNGjlypO3m+FYikdC0adN0zz33SJKmTJmi3bt3a926dVqwYIHl1vnHk08+qccff1wbNmzQ+eefr127dqmlpUW1tbX0k0d8l4I788wzVVpa2m9mUnd3t2pqaiy1yh8WLlyo559/Xi+//LLOOuus5P01NTU6evSoenp60vYvtj7buXOnDh48qIsuukhlZWUqKyvTli1btGbNGpWVlSkajdJPksaNG9fvisPnnnuu9u/fL0nJvij27+Dtt9+upUuX6uqrr9bkyZP13e9+V4sXL1Zra6sk+skLvgtA5eXlmjp1qtra2pL3JRIJtbW1qbGx0WLL7DHGaOHChXrqqaf00ksvqaGhIe3xqVOnasSIEWl91tHRof379xdVn1122WV68803tWvXruQ2bdo0zZ8/P/kz/STNmjWr3zT+PXv26Oyzz5YkNTQ0qKamJq2f4vG4tm3bVlT99PHHH/e7mmdpaakSiYQk+skTtmdBDGTjxo0mHA6bxx57zLz11lvmxhtvNKNHjzZdXV22m2bFzTffbCKRiPnrX/9q3n///eT28ccfJ/e56aabTH19vXnppZfMjh07TGNjo2lsbLTYan9InQVnDP1kzPEp6mVlZWblypVm79695vHHHzennXaa+eMf/5jcZ9WqVWb06NHmmWeeMf/85z/NFVdcUXTTixcsWGA+//nPJ6dh//nPfzZnnnmmueOOO5L70E/Z8WUAMsaYBx980NTX15vy8nIzffp08+qrr9pukjWSBtzWr1+f3OeTTz4xt9xyiznjjDPMaaedZr75zW+a999/316jfcIZgOin45577jkzadIkEw6HzcSJE80jjzyS9ngikTDLly830WjUhMNhc9lll5mOjg5LrbUjHo+bRYsWmfr6ejNy5EjzhS98wfzkJz8xvb29yX3op+xwPSAAgBW+qwEBAIoDAQgAYAUBCABgBQEIAGAFAQgAYAUBCABgBQEIAGAFAQgAYAUBCABgBQEIAGAFAQgAYMX/B3/UDYAY1eUjAAAAAElFTkSuQmCC)
```python
    img.shape
```
(3, 100, 100)

* define the number of the training samples
```python
    size = 2
    total_sample_size = 200
```
* turn the image to the np.array
```python
    def get_data(size, total_sample_size):
        #read the image
        image = read_image('C:\\Users\\jimore\\Downloads\\dataset\\have\\1.jpg')
        #reduce the size
        #get the new size
        dim1 = image.shape[1]
        dim2 = image.shape[2]
    
        count = 0
    
        #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
        x_geuine_pair = np.zeros([total_sample_size, 2, 3, dim1, dim2])  # 2 is for pairs
        y_genuine = np.zeros([total_sample_size, 1])
    
        for i in range(2):
            for j in range(int(total_sample_size/2)):
                ind1 = 0
                ind2 = 0
    
                #read images from same directory (genuine pair)
                while ind1 == ind2:
                    ind1 = np.random.randint(20)
                    ind2 = np.random.randint(20)
    
                # read the two images
                img1 = read_image('C:\\Users\\jimore\\Downloads\\dataset\\' + classes[i] + '\\' + str(ind1 + 1) + '.jpg')
                img2 = read_image('C:\\Users\\jimore\\Downloads\\dataset\\' + classes[i] + '\\' + str(ind2 + 1) + '.jpg')
    
    
                #reduce the size
                print(img1.shape)
    
                #store the images to the initialized numpy array
                x_geuine_pair[count, 0, :, :, :] = img1
                x_geuine_pair[count, 1, :, :, :] = img2
    
                #as we are drawing images from the same directory we assign label as 1. (genuine pair)
                y_genuine[count] = 1
                count += 1
    
        count = 0
        x_imposite_pair = np.zeros([total_sample_size, 2, 3, dim1, dim2])
        y_imposite = np.zeros([total_sample_size, 1])
    
        for i in range(int(total_sample_size/2)):
            for j in range(2):
    
                #read images from different directory (imposite pair)
                while True:
                    ind1 = np.random.randint(2)
                    ind2 = np.random.randint(2)
                    if ind1 != ind2:
                        break
    
                img1 = read_image('C:\\Users\\jimore\\Downloads\\dataset\\' + classes[ind1] + '\\' + str(j + 1) + '.jpg')
                img2 = read_image('C:\\Users\\jimore\\Downloads\\dataset\\' + classes[ind2] + '\\' + str(j + 1) + '.jpg')
    
    
                x_imposite_pair[count, 0, :, :, :] = img1
                x_imposite_pair[count, 1, :, :, :] = img2
                #as we are drawing images from the different directory we assign label as 0. (imposite pair)
                y_imposite[count] = 0
                count += 1
    
        #now, concatenate, genuine pairs and imposite pair to get the whole data
        X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
        Y = np.concatenate([y_genuine, y_imposite], axis=0)
    
        return X, Y
```
* load the image data to the $X$ ,load the label of the data to the $Y$
```python
    X, Y = get_data(size, total_sample_size)
```
* reshape the size of the data for the training process
```python
    X.shape
    X=X.reshape((400,2,100,100,3))
    X.shape
```
* seperate the `train_data` and the `test_data`
```python
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)
```
* define the training network
```python
    def build_model(input_shape):
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        #build convnet to use in each siamese 'leg'
        convnet = Sequential()
        convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                           kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128,(7,7),activation='relu',
                           kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4)))
        convnet.add(Flatten())
        convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3)))
        #encode each of the two inputs into a vector with the convnet
        encoded_l = convnet(left_input)
        encoded_r = convnet(right_input)
        #merge two encoded inputs with the l1 distance between them
        L1_distance = lambda x: K.abs(x[0]-x[1])
        both = Lambda(L1_distance, output_shape=lambda x: x[0])([encoded_l,encoded_r])
        prediction = Dense(1,activation='sigmoid')(both)
        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
        #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
        return siamese_net

    img_1 = x_train[:, 0]
    img2 = x_train[:, 1]

    network=build_model(input_dim)
```
* compile the train network
```python
    Adam = adam.Adam(0.00006)
    # optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
        #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    network.compile(loss="binary_crossentropy",optimizer=Adam)
```
* train the model
```python
    a=network.fit([img_1, img2], y_train, validation_split=.25,
          batch_size=20, verbose=2, epochs=10)
```
* save the checkpoints file
```python
    network.save('point.h5')
```
* check the accuracy of the model
```python
    pred = network.predict([x_test[:, 0], x_test[:, 1]])
    c=0
    for i in range(len(pred)):
        if pred[i][0]>0.5 and y_test[i][0]==1:
            c+=1
        if pred[i][0]<0.5 and y_test[i][0]==0:
            c+=1
    print(c/len(pred))
```
* predict our images
```python
    test="C:\\Users\\jimore\\Downloads\\1 (160).jpg"
    
    #导入需要检测的图片
    
    path='C:\\Users\\jimore\\Downloads\\下载.jpg'
    
    
    test=read_image(test)
    plt.imshow(test.reshape(100,100,3))
    plt.show()
    w,h=test.shape[1],test.shape[2]
    test=test.reshape(1,w,h,3)
```
* define a function to make sure of the label of the input_image with the similarity score between the input_image and the labeled images
```python
    def score(test,path,no):
        ans=0
        log_path=path+classes[no-1]
        log=os.listdir(log_path)
        for i in log[:20]:
            if i.endswith('jpg'):
                img_path=log_path+"\\"+i
                img=read_image(img_path)
                w,h=img.shape[1],img.shape[2]
                img=img.reshape(1,w,h,3)
                pre=network.predict([test,img])
                print(pre)
                ans+=pre[0][0]
        return ans/20
```
* predict the label of the input_image
```python
    MIN=1000000
    pos=0
    for i in range(20):
        if MIN>score(test,'C:\\Users\\jimore\\Downloads\\dataset\\',2):
            MIN=score(test,'C:\\Users\\jimore\\Downloads\\dataset\\',2)
            pos=i
    print("Category:"+classes[pos])
```
