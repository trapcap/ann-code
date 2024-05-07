# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:46:40 2024

@author: John
"""

import numpy


def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)
import numpy as np
from astropy.table import Table
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set the seed to a constant value
np.random.seed(42)

mpl.rcParams['axes.grid'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 15.0
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.minor.size'] = 10.0
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.minor.size'] = 10.0
mpl.rcParams['ytick.major.size'] = 15.0
mpl.rcParams['axes.labelsize'] = 35
mpl.rcParams['font.size'] = 35
mpl.rcParams["figure.figsize"] = (19,12)

mpl.rcParams["hist.bins"] = 200

# Converting from .fits to pandas dataframe
temp_table = Table.read('MICE_GAMA.fits', format='fits')
df = temp_table.to_pandas()

# Filtering simulated data to exclude magnitudes above 19.87
df_filtered = df[df['sdss_r_obs_mag'] <= 19.87]

spec_sample = df_filtered.sample(frac=(137/780), random_state=42)
photo_sample = df_filtered

z_true = spec_sample["z_cgal_v"]
z_b = spec_sample["Z_B"]

# u-band anomaly
uband_anomaly = spec_sample[spec_sample['sdss_u_obs_mag'] > 25]

# Filtering u-band anomaly
spec_sample = spec_sample[spec_sample['sdss_u_obs_mag'] < 25]

# Create a new dataframe excluding spec_sample- this will be test sample
test = df_filtered.drop(spec_sample.index)

# Create validation dataset from test sample (10% of test)
validation = test.sample(frac=0.1, random_state=42)

# Update test sample to ignore validation data to prevent bias during training
test = test.drop(validation.index)

# Filtering test sample based on 'sdss_u_obs_mag' column
test = test[test['sdss_u_obs_mag'] < 25]


# z_cgal_v will be treated as "true" redshift or y
z_cgal_v = test["z_cgal_v"].values
y_true = z_cgal_v

# Drop ra, dec, recal_wewight and Z_B columns
test = test.drop(['ra_gal_mag', 'dec_gal_mag', 'Z_B','recal_weight'], axis=1)

T_test = test.transpose()

X = T_test.values

SAMPLE = df_filtered

SAMPLE_filtered = df_filtered[df_filtered['sdss_u_obs_mag'] < 25]

# Designate desired variables and input variables
TargetVariable = ['z_cgal_v']
Predictors = ['sdss_u_obs_mag', 'sdss_g_obs_mag', 'sdss_r_obs_mag',
       'sdss_i_obs_mag', 'sdss_z_obs_mag', 'des_asahi_full_y_obs_mag',
       'vhs_j_obs_mag', 'vhs_h_obs_mag', 'vhs_ks_obs_mag']

X = SAMPLE_filtered[Predictors].values
y = SAMPLE_filtered[TargetVariable].values

from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)
 
# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(137/780), random_state=42)


"""
units = n: creating a layer with n neurons

input_dim = m : This means there are m predictors in the input data which is expected by the first layer. 
If you see the second dense layer, we don’t specify this value, because the Sequential model passes this information further to the next layers.

activation = ’relu’: The activation function used (e.g. Sigmoid, tanh, ReLU, etc)

batch_size = i: Number of rows passed through network in one go

Epochs=50: The same activity of adjusting weights continues for 50 times
"""


from keras.models import Sequential
from keras.layers import Dense
 
# create ANN model
model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same!
model.add(Dense(units=200, input_dim=9, kernel_initializer='normal', activation='relu'))
 
# Defining the second layer of the model
model.add(Dense(units=200, kernel_initializer='normal', activation='tanh'))
 
# The output neuron is a single fully connected node - (single calculated value / redshift)
model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')


# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_test, y_test):
    
    # Defining the list of hyper parameters to try
    batch_size_list=[ 50,  54,  58,  62,  66,  70,  74,  78,  82,  86,  90,
            94,  98, 102, 106, 110, 114, 118, 122, 126, 130, 134,
           138, 142, 146, 150, 154, 156, 162, 166, 170, 174, 178,
           182, 186, 190, 194, 198, 202, 206, 210, 214, 218, 222,
           226, 230, 234, 238, 242, 246, 250]
    epoch_list  =   [25]
    
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
    
    # initializing the trials
    TrialNumber=0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=200, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
 
            # Defining the Second layer of the model
            model.add(Dense(units=200, kernel_initializer='normal', activation='relu'))
 
            # The output neuron is a single fully connected node 
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal'))
 
            # Compiling the model
            model.compile(loss='mean_squared_error', optimizer='adam')
 
            # Fitting the ANN to the Training set
            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
 
            MAPE = np.mean(100 * (np.abs(y_test-model.predict(X_test))/y_test))
            Y_DATA = []
            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
            
            Y_DATA = Y_DATA.append(100-MAPE)
            print(Y_DATA)
            #SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
                                                                    #columns=['TrialNumber', 'Parameters', 'Accuracy'] ))
    #return(SearchResultsData)

# Calculate residuals & plot (mean average residuals is MAPE)

ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)
"""
Y_DATA = [112.52285836577617,113.66997113677805,116.19810120247504,111.65906349327184,
          110.13462859206311,114.06802376075245,117.51543391342365,108.3981607457963,
          117.69946000878822,115.80479448291298,111.29082715860612,116.00247024192593,
          117.91586697893968,114.19559343656299,117.77833248158873,118.3492289274115,
          117.41230751954897,119.09658165590460,118.56668363078245,118.36412726085833,
          119.94637279710064,119.44688222051998,122.03199456405748,124.01371971185847,
          119.66373901305973,121.58138083142839,118.78616759197334,117.74802900195247,
          124.62014753530572,120.81375030843968,121.21199932371786,114.17668365928125,
          119.24562321517685,121.43366736643576,127.11234835248634,124.85416827910257,
          129.44162280090114,124.73443580777395,126.87529450116077,124.8027133887887,
          124.10459859935652,126.17405812289944,125.81028094153032,124.66137646196438,
          120.00713917197062,123.31617957901793,126.0165857778284,123.71194399926445,
          129.44244559819364,124.72640295324408,120.4796858444089]

X_DATA =[ 50,  54,  58,  62,  66,  70,  74,  78,  82,  86,  90,
        94,  98, 102, 106, 110, 114, 118, 122, 126, 130, 134,
       138, 142, 146, 150, 154, 156, 162, 166, 170, 174, 178,
       182, 186, 190, 194, 198, 202, 206, 210, 214, 218, 222,
       226, 230, 234, 238, 242, 246, 250]

plt.plot(X_DATA,Y_DATA,'x',markersize=10)

# Linear regression to fit data

P = np.polyfit(X_DATA,Y_DATA,1)

x_fit = np.linspace(min(X_DATA),max(X_DATA),100)
y_fit = P[0]*x_fit + P[1]

plt.plot(x_fit, y_fit)
plt.ylabel('100 - MAPE')
plt.xlabel('batch size')
plt.legend(["Data","fit"])
"""
"""
X_DATA =[ 50,  54,  58,  62,  66,  70,  74,  78,  82,  86,  90,
        94,  98, 102, 106, 110, 114, 118, 122, 126, 130, 134,
       138, 142, 146, 150, 154, 156, 162, 166, 170, 174, 178,
       182, 186, 190, 194, 198, 202, 206, 210, 214, 218, 222,
       226, 230, 234, 238, 242, 246, 250]

Y_bright = np.array([76.56967217371222,79.22638436766977,71.37736661340573,80.12903581674799,
                        78.45139731975006,74.57480837273712,76.94586943328086,73.35841211914925,
                        76.17864526845429,80.58737923537166,81.48447486757127,74.38737072272843,
                        78.27624418733531,78.82102592991353,80.51047096423245,74.33894659822855,
                        77.57825204333895,82.84773572523777,74.38172038891466,76.7399012345287,
                        75.53768802680662,77.70163357059306,79.75568058145693,77.8650591726033,
                        76.85563682876709,76.23595770978622,76.16598341891913,75.13529691894885,
                        72.72317255971824,74.11160138486534,76.10846518284846,82.82668294196513,
                        71.43930482015344,73.37111031787053,78.25558952613768,75.50078073884954,
                        73.99036839335815,78.85326837208443,74.059607051834,78.38339723211197,
                        73.87322033219722,78.91419901426079,72.72919152128466,83.30068621116385,
                        78.01101962135549,74.96319038282127,82.26219166437434,81.8407496021742,
                        85.10404860895157,76.96526973579259,78.26501908601904])
plt.plot(X_DATA,Y_bright,'x',markersize=10)
    
P = np.polyfit(X_DATA,Y_bright,1)

x_fit = np.linspace(min(X_DATA),max(X_DATA),100)
y_fit = P[0]*x_fit + P[1]

plt.plot(x_fit, y_fit)

plt.ylabel('100 - MAPE')
plt.xlabel('batch size')
plt.legend(["Data","fit"])
#150 -50"""

