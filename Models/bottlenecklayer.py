def bottleneck():
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
    model.compile(loss="mean_squared_error",optimizer=tf.optimizers.Adam(learning_rate=0.01))
    return model
inverted_adaboost=bottleneck()

