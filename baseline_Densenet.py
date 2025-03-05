import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Flatten
def base_nn():
    model=Sequential()
    model.add(Dense(150,input_dim=x_train_scaled.shape[1],activation="relu",kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(150,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(100,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(123,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1,activation="linear"))
    model.compile(loss="mean_squared_error",optimizer=tf.optimizers.Adam(learning_rate=0.001))
    return model

model=base_nn()
model.summary()


model.fit(x_train_scaled,y_train,epochs=50,batch_size=32,validation_data=(x_val_scaled,y_test),verbose=1)


pred=model.predict(x_val_scaled)
print(f"Root mean squared error when training with neural network:{np.sqrt(mean_squared_error(y_test,pred))}")
