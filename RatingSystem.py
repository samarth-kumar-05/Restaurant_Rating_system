# Restaurent Rating System

# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')

#Importing the dataset
import pandas as pd 
import numpy as np
ZomatoData = pd.read_csv('ZomatoData.csv')
print('Shape of the data',ZomatoData.shape)

ZomatoData = ZomatoData.drop_duplicates()
print('Shape of data after removing duplicate value',ZomatoData.shape)

#printing first few rows from dataset
ZomatoData.head(10)
                         
#Study of target variable

# import matplotlib.pyplot as plt
# plt.hist(ZomatoData['Rating'])
# plt.show()

#Data Exploration
ZomatoData.info()
#checking unique values for each column
ZomatoData.nunique()

# =============================================================================
# '''By looking at the data we can reject Restaurant ID, Restaurant Name , City ,Cuisines,Locality Verbose , locality , Cuisines '''
# =============================================================================

#we can however use the number of cuisines offered so 
#funtion to count number of cuisines
def cuisine_counter(inp):
    NumCuisines  = len(str(inp).split(','))
    return NumCuisines
ZomatoData['CuisineCount'] = ZomatoData['Cuisines'].apply(cuisine_counter)

#Removing Useless Columns from th data
UselessColumns = ['Restaurant ID', 'Restaurant Name','City','Address','Locality', 'Locality Verbose','Cuisines']
ZomatoData = ZomatoData.drop(UselessColumns,axis=1)


# =============================================================================
# by looking at the bar charts in this data, "Country Code", "Currency", "is delivering now" and "Switch to order menu" are too skewed. There is just one bar which is dominating and other categories have very less rows or there is just one value only. Such columns are not correlated with the target variable because there is no information to learn. The algorithms cannot find any rule like when the value is this then the target variable is that.
#only these columns are selected 'Has Table booking', 'Has Online delivery', 'Price range'                   
# =============================================================================

#Outlier Treatement

# =============================================================================
# on checking the votes column we can see that there are otliers so we treat them by replacing any value above 4000 by a different value
# =============================================================================
ZomatoData['Votes'][ZomatoData['Votes']<4000].sort_values(ascending=False)
#we see the nearest value is 3986
ZomatoData['Votes'][ZomatoData['Votes']>4000] = 3986

# Finding nearest values to 50000 mark
ZomatoData['Average Cost for two'][ZomatoData['Average Cost for two']<50000].sort_values(ascending=False)

#replacing the oulier with 8000
ZomatoData['Average Cost for two'][ZomatoData['Average Cost for two']>50000] =8000

# =============================================================================
# feature selection
# =============================================================================

#calculating the correlation coefficient for continuous data
# Calculating correlation matrix
ContinuousCols=['Rating','Longitude', 'Latitude', 'Votes', 'Average Cost for two']

# Creating the correlation matrix
CorrelationData=ZomatoData[ContinuousCols].corr()
CorrelationData

# Filtering only those columns where absolute correlation > 0.2 with Target Variable
CorrelationData['Rating'][abs(CorrelationData['Rating']) > 0.2]
#so final selected continuous columns are Votes and Average Cost for two

# =============================================================================
# performing ANOVA test for categorical data
# =============================================================================
# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    print('##### ANOVA Results ##### \n')
    for predictor in CategoricalPredictorList:
        CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    
    return(SelectedPredictors)
# Calling the function to check which categorical variables are correlated with target
# Calling the function to check which categorical variables are correlated with target
CategoricalPredictorList=['Has Table booking', 'Has Online delivery', 'Price range']
FunctionAnova(inpData=ZomatoData, 
              TargetVariable='Rating', 
              CategoricalPredictorList=CategoricalPredictorList)
#final we select 'Has Table booking', 'Has Online delivery', 'Price range' for categorical data columns

# =============================================================================
# selecting final predictors for our model
# =============================================================================
SelectedColumns = ['Votes','Average Cost for two','Has Table booking','Has Online delivery','Price range']

#selecting the final columns 
DataForML = ZomatoData[SelectedColumns]

#saving this data for refrence
DataForML.to_pickle('DataForML.pkl')

# =============================================================================
# Converting the binary nominal variable to numeric 1/0 mapping
# =============================================================================
DataForML['Has Table booking'].replace({'Yes':1, 'No':0}, inplace=True)
DataForML['Has Online delivery'].replace({'Yes':1, 'No':0}, inplace=True)

# =============================================================================
# Converting the nominal variable to numeric using get_dummies()
# =============================================================================
DataForML_Numeric = pd.get_dummies(DataForML)
DataForML_Numeric['Rating'] = ZomatoData['Rating']

# =============================================================================
# Splitting the data into test and train set
# =============================================================================
TargetVariable='Rating'
Predictors=['Votes', 'Average Cost for two', 'Has Table booking',
           'Has Online delivery', 'Price range']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.3,random_state=428)

# =============================================================================
# ### Sandardization of data ###
# =============================================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================================================
# Fitting  Decision Trees
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
RegModel = DecisionTreeRegressor(max_depth=10,criterion='mse')

DT = RegModel.fit(X_train,y_train)
prediction = DT.predict(X_test)

from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, DT.predict(X_train)))


###########################################################################
print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())

# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['Rating']-TestingDataResults['PredictedRating']))/TestingDataResults['Rating'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# =============================================================================
# choosing only the most important variable
# =============================================================================
#we select only 'Votes', 'Average Cost for two', 'Price range' for final model

TargetVariable = 'Rating'
Predictors = ['Votes','Average Cost for two','Price range']

X = DataForML_Numeric[Predictors].values
y = DataForML_Numeric[TargetVariable].values
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# =============================================================================
# retraining the model with 100% data
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
RegModel = DecisionTreeRegressor(max_depth=6,criterion='mse')
FinalDecisionTreeModel = RegModel.fit(X, y)

import pickle
import os

# Saving the Python objects as serialized files can be done using pickle library
# Here let us save the Final ZomatoRatingModel
with open('FinalDecisionTreeModel.pkl', 'wb') as fileWriteStream:
    pickle.dump(FinalDecisionTreeModel, fileWriteStream)
    # Don't forget to close the filestream!
    fileWriteStream.close()
    
print('pickle file of Predictive Model is saved at Location:',os.getcwd())
# This Function can be called from any from any front end tool/website
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
    X=PredictorScalerFit.transform(X)
    
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
    return(Predictions.to_json())

# Function call
FunctionGeneratePrediction(314,1100,3)

filename = 'PredictorScaler.pkl'
pickle.dump(PredictorScaler,open(filename,'wb'))