import matplotlib.pyplot as plt
import numpy as np

path=r"D:\machine learning\exercise_dataset\ex1data1.txt"
with open(path,"r") as ex1data1:
    contents=ex1data1.readlines()
    
population=[]
profit=[]
x=[]
cost=[]
for i in range(len(contents)):
    contents[i]=contents[i].rstrip().split(",")
    population.append(float(contents[i][0]))
    profit.append([float(contents[i][1])])
    x.append([1,float(contents[i][0])])

figure1=plt.figure()
figure1.add_subplot(221)
plt.scatter(population,profit,c="r",marker="x",label="Trainning Data")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")


x=np.matrix(x)
y=np.matrix(profit)
theta=np.matrix(np.array([0.,0.]))

def compute_cost(x,y,theta):
    j=np.sum(np.power(x*theta.T-y,2))/(2*len(x))
    return cost.append(j)

def gd(x,y,alpha,theta,iterations):
    for i in range(iterations):
        inner=x*theta.T-y
        theta[0,0]=theta[0,0]-alpha*np.sum(np.multiply(inner,x[:,0]))/(len(x))#multiply为对应相乘，非矩阵相乘*、dot
        theta[0,1]=theta[0,1]-alpha*np.sum(np.multiply(inner,x[:,1]))/(len(x))
        compute_cost(x,y,theta)
    return theta

alpha=0.01
iterations=1000
print(gd(x,y,alpha,theta,iterations))

x_lr=np.linspace(x[:,1].min(),x.max(),10000)
y_lr=theta[0,0]+theta[0,1]*x_lr
plt.plot(x_lr,y_lr,"b",label="Linear Regression")
plt.legend(loc="best")
plt.yticks(range(-5,26,5))
plt.xticks(range(4,25,2))

figure1.add_subplot(224)
plt.plot(range(0,1000,1),cost)

plt.show()
