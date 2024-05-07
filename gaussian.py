# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:03:19 2024

@author: jdjac
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

b_tan50 = [94.96511349769533,95.04391807104919,94.81507839832749,
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
          95.90760327410176]

L = ['sigmoid_100','sigmoid_50','relu','leaky_relu','tanh']
C = ['purple','blue','red','green','orange']

def gaussian_pdf(x, mu, sigma):
    coefficient = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * np.exp(exponent)


def plot(x,labelnumber):
    mu = np.mean(x)
    sigma = np.std(x)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma),label=L[labelnumber],color=C[labelnumber],linewidth=3)
    plt.legend()


plot(b_sig100,0)
plot(b_sig50,1)
plot(b_relu,2)
plot(b_leaky,3)
plot(b_tan50,4)
plt.xlabel('accuracy / %')
plt.ylabel('PDF')


