# hello-world-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
import pylab as pl

np.random.seed(0)
x=np.random.random(size=(20,1))
y=3*x.squeeze()+2+np.random.randn(20)
plt.plot(x.squeeze(),y,"o")

lr=LinearRegression()
lr.fit(x,y)
x_fit=np.linspace(0,1,100)[:,np.newaxis]
y_fit=lr.predict(x_fit)
plt.plot(x_fit,y_fit,"-")
plt.show()
