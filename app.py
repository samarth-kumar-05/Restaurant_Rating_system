import pandas as pd
from flask import Flask,escape,request
from flask.templating import render_template
import pickle

PredictorScaler = pickle.load(open("PredictorScaler.pkl",'rb'))

def FunctionPredictResult(InputData):
    import pandas as pd
    Num_Inputs=InputData.shape[0]
    
    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input
    
    # Appending the new data with the Training data
    DataForML=pd.read_pickle('DataForML.pkl')
    InputData=InputData.append(DataForML)
    
    # Generating dummy variables for rest of the nominal variables
    InputData=pd.get_dummies(InputData)
            
    # Maintaining the same order of columns as it was during the model training
    Predictors=['Votes', 'Average Cost for two', 'Price range']
    
    # Generating the input values to the model
    X=InputData[Predictors].values[0:Num_Inputs]
    
    # Generating the standardized values of X since it was done while model training also
    X=PredictorScaler.transform(X)
    
    # Loading the Function from pickle file
    import pickle
    with open('FinalDecisionTreeModel.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()
            
    # Genrating Predictions
    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    return(PredictionResult)
# Calling the function for new sample data
NewSampleData=pd.DataFrame(
data=[[314,1100,3],
     [591,1200,4]],
columns=['Votes', 'Average Cost for two', 'Price range'])

print(NewSampleData)

# Calling the Function for prediction
FunctionPredictResult(InputData= NewSampleData)

# Creating the function which can take inputs and return predictions
def FunctionGeneratePrediction(inp_Votes, inp_Average_Cost, inp_Price_range):
    
    # Creating a data frame for the model input
    SampleInputData=pd.DataFrame(
     data=[[inp_Votes , inp_Average_Cost, inp_Price_range]],
     columns=['Votes', 'Average Cost for two', 'Price range'])

    # Calling the function defined above using the input parameters
    Predictions=FunctionPredictResult(InputData= SampleInputData)

    # Returning the prediction
    return(Predictions)

app = Flask(__name__)


@app.route('/')
def home():

    return render_template("index.html")
@app.route('/predict' , methods = ["GET","POST"])
def prediction():
    if request.method == "POST":
        try:
            votes = str(request.form['votes'])
            cost = str(request.form['cost'])
            range = str(request.form['range'])
            print(votes,cost,range)
            calculated = FunctionGeneratePrediction(int(votes),int(cost),int(range))
            predicted = (calculated.to_string(index=False))
            predicted = predicted[11:]
            print(predicted)
            return render_template('final.html',predictedd = "The Final Rating is : {}".format(predicted))
        except Exception as e:
            return('Something is wrong '+ str(e))
    else:
        return render_template("prediction.html")
@app.route("/result")
def last():
    return render_template('final.html')

@app.route('/aboutus')
def about():
    return render_template('aboutus.html')


if(__name__ == '__main__'):
    app.run()