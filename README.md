# Heroku-Iris-Dataset
API for Iris Dataset


Iris.py contains data preprocessing and training Logistic Regression model for predicting category of flower.

LR_Regularization_Dropout_Adam.py is a Logistic Regression model built from scratch using numpy.
It is the extension of the assignment provided by Dr. Andrew Ng in his Specialization Course  
You can use L1/L2 regularization, Relu activation for hidden layers, Sigmoid/Softmax for final layer, Dropout.

For training use: 

        L_layer_model(X, y,Output_classes, layers_dims=[X.shape[1],6,6,np.unique(y).shape[0]], 
                      predict_result=False,activation_type="multiclass", reg_type="l2",keep_prob=0.7, 
                      mini_batch_size=10, n=1, learning_rate = 0.001,lambd=0.01, num_epochs =500)

                       
                       
params: 

       1)layers_dims = List contains number of neurons in one respective layer
                      and [len(layer_dims) - 1] gives L Layer Neural Network               
       2)activation_type = The activation to be used in this layer, stored as a text string: "bianry" or "multiclass"
       3)reg_type = Type of regularization to use "l1" or "l2"
       4)keep_prob = Percentage of neurons to be kept active 
       5)learning_rate = learning rate of the gradient descent update rule
       6)n = 1 or 2, used for random initialization of weights, when 
             n = 1, we get LeCun Initializer
             n = 2, we get He Initializer
       7)lambd = Regularization parameter, int
       8)num_epochs = number of epochs
       9)predict_result = False while training, True when predicting the ground truth 
                          values (False only when ground truth values are present)
                          Must be kept False if you have ground truth values
                          while predicting
                       
After training, the model will be saved as pickled file, the saved model name is Logisitic_Regression.pkl
This pickled model is then use for prediction.
You need flask for web application.

Following are the steps to be followed:
1) Create an account on Heroku and install flask
2) Edit the Index.html template, it’s the html file which flask renders for web application.
3) Make changes to the app.py file, in this file use the pickled model to perform 
   predictions on the data provided by user on the Heroku or web server(local machine).
4) Using command prompt run the command “flask run” to check if the model is working on web(local machine).
5) Generate requirement.txt file which lets the Heroku know what are the dependencies and requirements for the project. 
   Also, Procfile is required which tells Heroku what kind of app we are trying to run and how to serve it to the end user.
6) Link to Heroku api: https://iris-model-api.herokuapp.com/
