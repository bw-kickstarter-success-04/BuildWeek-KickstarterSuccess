from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib

DB = SQLAlchemy()
APP = Flask(__name__)

APP.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sqlite3'
APP.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

DB.init_app(APP)
# Set up the main route
@APP.route('/')
def main():
        return render_template('home.html')
@APP.route('/about')
def about():
        return render_template('about.html')

@APP.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == 'POST':
        # Get form data
        goal = request.form.get('goal')
        month = request.form.get('month')
        year = request.form.get('year')
        duration = request.form.get('duration')
        currency = request.form.get('currency')
        country = request.form.get('country')
        main_category = request.form.get('main_category')
        time_since_last_project = request.form.get('time_since_last_project')
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(goal, month, year, duration, currency, 
                                                  country, main_category, time_since_last_project)
            
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
    pass
pass
        
def preprocessDataAndPredict(goal, month, year, duration, currency, 
                             country, main_category, time_since_last_project):
    
    #keep all inputs in array
    test_data = [goal, month, year, duration, currency, 
                 country, main_category, time_since_last_project]
                 
    print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array
    test_data = test_data.reshape(1,-1)
    print(test_data)
    
    # Open and Load Pickle file 
    file = open("kickstarterapp/model.pkl","rb")
    model = joblib.load(file)
    #predict
    prediction = model.predict(test_data)

    print(prediction)
    return prediction
pass
    


        
if __name__ == '__main__':
    APP.run()