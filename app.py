import numpy as np
import flask
from flask import render_template
from Iris import Output_classes,y_test
from LR_Regularization_Dropout_Adam import api_prediction
import pickle

app = flask.Flask(__name__, template_folder='templates')
model_to_pickle = "Logisitic_Regression.pkl"
with open(model_to_pickle, 'rb') as file:
    pickled_model = pickle.load(file)


@app.route('/',methods=['GET','POST'])
def main():
    '''
    For rendering results on HTML GUI
    '''
    if flask.request.method == 'GET':
        return (render_template('index.html'))
    
    if flask.request.method == 'POST':
        test_features = [float(x) for x in flask.request.form.values()]
        final_features = np.array(test_features)
        final_features = final_features.reshape((1,final_features.shape[0]))
    
        
        prediction = api_prediction(final_features,pickled_model, y_test,
                      Output_classes,keep_prob=1,predict_result=True, 
                      activation_type="multiclass" ,flags="predict_y")
    
        prediction = np.squeeze(prediction).astype(int)
        #print(prediction)
        if prediction==0: pred="Iris Setosa"
        elif prediction==1: pred="Iris Versicolor"
        else: pred="Iris Virginica" 
        #output = round(prediction[0], 2)
    
        return render_template('index.html', prediction_text='Given flower features belongs to {} flower'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)