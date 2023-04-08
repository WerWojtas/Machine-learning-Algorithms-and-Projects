#basic linear regression algorithm

import matplotlib.pyplot as plt
import pandas as pd
T_points=pd.read_csv(r'C:\Python\ML\Linear_regression\train.csv')



def new(T_points,weight,bias):
    n=len(T_points)
    a=0
    b=0
    for i in range(n):
        x=T_points.iloc[i].x
        y=T_points.iloc[i].y
        a+=(-2/n)*x*(y-(weight*x+bias))
        b+=(-2/n)*(y-(weight*x+bias))
    return a,b


def linear_regression(T_points,learning_rate,operations):
    weight=0
    bias=0
    for i in range(operations):

        a,b=new(T_points,weight,bias)
        if i%50==0:
            print(i,a,b)
        weight=weight-learning_rate*a
        bias=bias-learning_rate*b
    return weight, bias




weight,bias=linear_regression(T_points,0.0001,300)
print(weight,bias)


plt.scatter(T_points.x, T_points.y, color="pink")

plt.plot(list(range(100)),[weight*x+bias for x in range(100)], color="black")



plt.show()
