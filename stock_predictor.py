#Stock Market App


#Load Stock Data
import quandl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd

bhel=quandl.get("XNSE/BHEL", authtoken="exsk8vMy75wfr5yTRycF")

#Filtering/Cleaning Data
print(bhel.columns)
print(bhel.shape)

#High/Low Percentage (Variance between daily high and low values)
bhel["HlPct"]=(bhel['High']-bhel['Low']) / bhel['Close'] * 100

#Percentage Change - Change in percentage in stock price from last closing price
bhel['PctChange']=(bhel['Close']-bhel['Open'])/bhel['Open']*100

bhel=bhel[ ['Close','HlPct','PctChange','Volume']]

#Features and Labels
bhel['FuturePrice']=bhel['Close'].shift(-7)

#Search for Missing Values (NaN)

print(bhel.isnull().any())
print(bhel.isnull().sum())

#Scaling Data

features=np.array(bhel.drop(['FuturePrice'],1))
features=preprocessing.scale(features)

features = features[:-7]           #First 2340 Records
predictions = features[-7:]         #Last 7 Records

#Save Prediction Data for Later Use
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("predictions.csv")

labels=np.array(bhel['FuturePrice'])
labels=labels[:-7]                 #First 2340 Records

#print(len(features))
#print(len(predictions))
#print(len(labels))

#Training and Testing Data Sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

#print(train_features.shape)
#print(test_features.shape)
#print(train_labels.shape)
#print(test_labels.shape)

#Model
#Define model
model = Sequential()
model.add(Dense(10,activation='relu', kernel_initializer='he_normal', input_shape=(4,)))
model.add(Dense(32,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mse')

#Train the model
model.fit(train_features, train_labels,epochs=10, batch_size=32, verbose=0)

#Evaluate the model
error = model.evaluate(test_features, test_labels, verbose=0)

#Save model for future 
model.save('stock_predictor.h5')


#Visualize the Information

for index in range(-7,0,1):
    print(index)
    rowIndex = bhel.iloc[-index].name
    bhel['FuturePrice'][index]=prediction_prices[index+7]
    
bhel=bhel.truncate(before='2018-12-21')
bhel['FuturePrice'].plot()
bhel['Close'].plot()
plt.legend()
plt.xlabel()
plt.ylabel('Price')
plt.show()

