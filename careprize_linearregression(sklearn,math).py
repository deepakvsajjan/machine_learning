import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression;
url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv";
df=pd.read_csv(url)
#print(df.head());
print(df.columns);
print(df.shape);
print(df.head());
print(df.info());
print(df.isnull().sum());
df.drop(df[df["horsepower"].isnull()].index,inplace=True)
plt.figure();
plt.scatter(df["horsepower"],df["mpg"]);
plt.show();
x=df["horsepower"].values;
y=df["mpg"].values;
x_mean=np.mean(x);
y_mean=np.mean(y);
numerator=np.sum((x-x_mean)*(y-y_mean));
denominator=np.sum((x-x_mean)**2);
w=numerator/denominator;
c=y_mean-w*x_mean
y_pred=w*x+c
print(c);
print(df[["mpg"]]);
plt.scatter(x,y,color="blue");
plt.plot(x,y_pred,color="red");
plt.show();
model=LinearRegression();
model.fit(x.reshape(-1,1),y.reshape(-1,1));
y_pred_sklearn=model.predict(x.reshape(-1,1));
print(y_pred_sklearn)
