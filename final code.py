# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:17:46 2024

@author: jdjac
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:18:04 2024

@author: jdjac
"""
# seeds used

seeds = [1,2,3,
         4,5,6,
         7,8,9,
         10]


import numpy


def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)
import numpy as np
from astropy.table import Table
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Need to fix RNG for reproducible results
import random
import tensorflow as tf

seed=42

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


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
# If fitting faint sample, use MICE_KiDS.fits
# If bright sample, use MICE_GAMA.fits

temp_table = Table.read('MICE_KiDS.fits', format='fits')
df = temp_table.to_pandas()

# Filtering simulated data to exclude magnitudes above 19.87
df_filtered = df[df['sdss_r_obs_mag'] <= 19.87]

spec_sample = df_filtered.sample(frac=(137/780), random_state=seed)
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
validation = test.sample(frac=0.1, random_state=seed)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(137/780), random_state=seed)


# importing the libraries
from keras.models import Sequential
from keras.layers import Dense
 
# create ANN model
model = Sequential()
 
# Defining the Input layer and FIRST hidden layer, both are same
model.add(Dense(units=200, input_dim=9, kernel_initializer='normal', activation='sigmoid'))
 
# Defining the second layer of the model
model.add(Dense(units=200, kernel_initializer='normal', activation='sigmoid'))
 
# The output neuron is a single fully connected node - (single calculated value / redshift)
model.add(Dense(1, kernel_initializer='normal'))
 
# Compiling the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the Training set
model.fit(X_train, y_train ,batch_size = 200, epochs = 100, verbose=1)
 
# Generating Predictions on testing data
Predictions=model.predict(X_test)
 
# Scaling the predicted data back to original scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test data back to original scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)
 
TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['Z_true']=y_test_orig
TestingData['Z_pred']=Predictions
TestingData.head()

# Computing the absolute percent error
APE=100*(abs(TestingData['Z_true']-TestingData['Z_pred'])/TestingData['Z_true'])
TestingData['APE']=APE
 
print('The Accuracy of ANN model is:', 100-np.mean(APE))
TestingData.head()

plt.hist(Predictions, label='z_sigmoid_100_epochs',color='purple',alpha=0.6)
plt.hist(y_test_orig,label='z_true',color='black',alpha=0.5)

plt.xlabel('z / redshift')
plt.ylabel('counts')

accuracy = 100 - np.mean(APE)

plt.legend()

# Bright sample accuracies

b_tan = [94.96511349769533,95.04391807104919,94.81507839832749,
         95.12982337123098,94.68402266827887,94.94194023062478,
         94.65648446107664,94.74285043050322,94.64300131853066,
         94.8167258460473] #

b_sig50 = [91.95496958514921,92.53668774105468,91.98772414366206,
           92.52758477140125,91.85739666528153,92.55459540748105,
           91.65909341295500,92.30531884600694,92.18555228522918,
           91.70118634440038] #

b_sig100 = [93.71784393518615,93.77150729909258,92.90972904296129,
            93.82028367222519,93.47382808393552,93.89956108353583,
            93.93151487554312,92.88129625159658,93.77349950118705,
            93.39852263919892] #

b_leaky = [95.79895754698981,95.52065143686383,95.19684563784182,
           95.98617427475122,94.77867567063363,95.44165376011375,
           95.24947181011575,95.57073953145684,95.60870321190188,
           95.39393324638311] #

b_relu = [95.92984813100826,95.82137077762175,95.76642509911994,
          96.01198489469033,95.68465890563513,95.84861919495717,
          95.3031589664163,95.87284832042678,95.90760327410176,
          95.90760327410176] # 


# Faint sample accuracies

f_tan50 = [93.79579311665213,92.53668774105468,94.19415953937803,
           94.33416873155066,94.10227032260117,94.25236621254932,
           94.33386215784766,93.92046276423739,94.33517025438390,
           94.33934325557826]

f_tan100 = [94.88941225531562,94.81697734184962,94.51708314391053,
            94.66330462529449,94.74912742012877,94.63675737168484,
            94.92421460986637,94.52013320506866,94.52013320506866,
            94.80385621801295]

f_sig50 = [90.61729811265856,91.704495161664,91.66498657055833,
           91.49578107188799,91.55780696380563,91.694191559632,
           91.80685076383891,91.64759670223755,92.13751862103543,
           91.69462287225807]

f_sig100 = [92.92190813343140,92.75113857029537,92.853184576889,
            92.47036166036285,92.89243348503112,92.98737210039155,
            92.81852465253755,92.61997745625567,93.09049567404108,
            92.94986864515289]

f_leaky = [95.00841345411378,95.07390896519190,94.55536857310100,
           94.61937695941373,95.06218335647111,95.03974524948997,
           95.27873140826603,94.65051352386473,95.19554248263147,
           95.26209558817237]

f_relu = [94.79192190827128,95.22974462962476,95.19211434668684,
          95.24409108753204,95.12527969022780,95.11996363269701,
          95.23640014387635,95.08997495894627,95.36035682325351,
          95.40106629658034]
