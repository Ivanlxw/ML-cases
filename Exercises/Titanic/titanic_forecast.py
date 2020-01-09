"""
Use training set to predict inputs in test set
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sys
from sklearn.model_selection import GridSearchCV


def clean_data(filepath, mode="train"):

    df = pd.read_csv(filepath)
    if mode == "train":
        inputs = df[['Pclass','Sex','Age', 'SibSp', 'Parch','Fare','Embarked','Survived']]
    elif mode == "test":
        inputs = df[['Pclass','Sex','Age', 'SibSp', 'Parch','Fare','Embarked']]

    inputs = pd.get_dummies(inputs, columns=['Pclass','Sex','Embarked'])

    if mode == "train":
        inputs = inputs.dropna()
        X = inputs.drop('Survived', axis=1)
        y = inputs['Survived']

        # train_test_split
        return train_test_split(X, y, test_size=0.33, random_state=41)
    elif mode == "test":
        inputs = inputs.fillna(method="pad")
        return inputs


X_train, X_test, y_train, y_test = clean_data("./train.csv")

def training(method="nn"):
    if method == "SVC":
        # gridsearch before actual training
        param_grid = {"C": [10,50,100,150], "gamma":[0.0001, 0.0005,0.001,0.01]}
        grid = GridSearchCV(SVC(), param_grid,verbose=2, scoring='accuracy',cv=10)

        grid.fit(X_train, y_train)
        predictions = grid.predict(X_test)

        # Evaluation
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test, predictions))

        return grid

    elif method == "RFC":
        """
        Using RandomForest
        Eval: Gets you 80-81% accuracy
        """
        from sklearn.ensemble import RandomForestClassifier
        param_grid = {"n_estimators":[40, 50, 55, 60, 65], "min_samples_split":[2,4,8,16,32]}
        grid = GridSearchCV(RandomForestClassifier(), param_grid,verbose=2, scoring='accuracy',cv=10)

        grid.fit(X_train, y_train)
        predictions = grid.predict(X_test)

        # Evaluation
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test, predictions))

        return grid

    elif method == "nn":
        """
        Using nn
        """
        import keras
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation
        from keras.optimizers import SGD,RMSprop, Adagrad, Adam

        lr = 0.01
        epochs= 1000
        optimizer = Adam(lr=lr)


        model = Sequential()
        model.add(Dense(32, input_dim=12))          #add input layer of 32 nodes with same input shape as training sample
        model.add(Activation('relu'))
        model.add(Dense(16))            #Second hidden layer has 512 nodes
        model.add(Activation('relu'))
        model.add(Dense(1))        #Adds fully connected output layer with nodes = n possible class labels
        model.add(Activation('sigmoid'))        #Adds softmax activation layer

        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        model.fit(X_train,y_train,epochs=epochs,verbose=2,batch_size=64)
        model.evaluate(X_test, y_test, batch_size=64, verbose=2)

        return model

def predict_test(method="nn"):
    test_data = clean_data("./test.csv","test")
    if method == "nn":
        model = training()
        #predict function should return an array
        test_pred = model.predict(test_data)
    elif method == "RFC":
        grid = training("RFC")
        test_pred = grid.predict(test_data)
    elif method == "SVC":
        grid = training("SVC")
        test_pred = grid.predict(test_data)

    df = pd.read_csv("./test.csv")
    df['Survived'] = test_pred
    df['Survived'] = df['Survived'].apply(lambda x: 1 if x > 0.5 else 0)

    return df



if __name__ == "__main__":
    """
    Model has already been pre-trained with input data initialized in the middle of the whole function portion above
    (Bad practice and should improve but works for now)
    """
    # Training data
    # training(sys.argv[1])

    #Testing data
    pred_survived = predict_test(sys.argv[1])           #nn performs best here so method=nn, though other methods have been written
    pred_survived.to_csv("./{}_test.csv".format(sys.argv[1]))
