import math
import numpy as np

'''
See : https://github.com/CoryMagnuson/QUANT-TRAINING/blob/master/Explicit%20Finite%20Differences%20Method%20-%20Option%20Valuation.ipynb
planetmath.org/solvingtheblackscholespdebyfinitedifferences
https://www.quantopian.com/posts/python-black-and-scholes-pde-finite-difference-method
'''

T = 0.5  #Time to Expiry in Years
E = 50  #Strike
r = .10  #Risk Free Rate
SIGMA = .40  #Volatility
Call = True   #True=Call False=Put
NAS = 200  #Number of Asset Steps - Higher is more accurate, but more time consuming

ds = 2 * E / NAS  #Asset Value Step Size
dt = (0.9/NAS/NAS/SIGMA/SIGMA)  #Time Step Size
NTS = int(T / dt) + 1  #Number of Time Steps
dt = T / NTS #Time Step Size
print("Asset Step Size %.2f Time Step Size %.2f Number of Time Steps %.2f Number of Asset Steps %.2f" %(ds, dt, NTS, NAS))

#Setup Empty numpy Arrays
value_matrix = np.zeros((int(NAS), int(NTS)))
asset_price = np.arange(0, NAS*ds, ds)

#Evaluate Terminal Value for Calls or Puts
if Call == True:
    value_matrix[:,-1]= np.maximum(asset_price - E,0)
else:
    value_matrix[:,-1]= np.maximum(E - asset_price,0)
    
#Set Boundary in Grid
for x in range(1,NTS):
    # the payoff discounted until time 0
    value_matrix[0,-x-1] = value_matrix[0,-x]* math.exp(-r*dt)
    # Upper boundary:
    value_matrix[-1,-x-1] = value_matrix[-1,-x]* math.exp(-r*dt)

#Set Mid and upper Values in Grid
for x in range(1,int(NTS)):
    for y in range(1,int(NAS)-1):
        #Evaluate Option Greeks
        Delta = (value_matrix[y+1,-x] - value_matrix[y-1,-x]) / 2 / ds
        Gamma = (value_matrix[y+1,-x] - (2 * value_matrix[y,-x]) + value_matrix[y-1,-x]) / ds / ds
        
        #Set Mid Values
        value_matrix[y,-x-1] = value_matrix[y,-x] + dt * (0.5 * SIGMA**2 * asset_price[y]**2 * Gamma) + dt * (r * asset_price[y] * Delta) - dt * (r * value_matrix[y,-x])
          
    