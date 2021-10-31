## FIFA19 SPORT PREDICTION ML
This is FIFA19 Project assignmnet that uses Machine Learn Model to predict the Player position based on the FIFA19 Datasets are deployed on production using Flask API

## LIVE DEMO
The project is deployed on : https://fifa19-player-prediction.herokuapp.com/

## NOTEBOOK
COLAB NOTEBOOK : https://colab.research.google.com/drive/1KTGPhRnEhs2KEpwRta3c6Ci2VCALG1LD?usp=sharing

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict Player positions based on trained  data in 'data.csv' file found in the project directory.
2. app.py - This contains all Flask APIs that receives players details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter Players attributes to be predicted and displays the predicted employee salary.



### Running the project
1. Navigate to the home directory and create a machine learning model by simply running the following command.

```
python model.py
```
IMPACT : This will create a serialized version of the model into a model.pkl file that will be used to interact with in the Flask app.


2. Then, Run the Flask API app by running the following command:
```
python app.py

```
IMPACT: This will open a Flask API app and run by default on PORT 5000.

3. Navigate to URL : http://localhost:5000

If everything goes as planned, you should be able to see the predicted Player position based on 10 Selected Paramenters .


4.  You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```



## AUTHOR
The project is created by : Zubeir Msemo
