import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path=r"D:\machine learning\exercise_dataset\ex1data2.txt"
with open(path,"r") as ex1data2:
    contents1=pd.read_csv(ex1data2,names=["Size","Bedrooms","Price"])

means=contents1.mean().values
stds=contents1.std().values
maxs=contents1.max().values
mins=contents1.min().values

contents=(contents1-means)/stds #特征归一化
contents1=np.array(contents1)
contents.insert(0,"ones",1)

contents=np.matrix(contents)
x=contents[:,0:contents.shape[1]-1]
y=contents[:,contents.shape[1]-1:contents.shape[1]]
theta=np.matrix(np.zeros(x.shape[1]))
costy=[]

def cost_func(x,y,theta):
    '''计算代价函数'''
    hx=x*theta.T
    cost=np.sum(np.power(hx-y,2))/(2*len(y))
    costy.append(cost)

def gd(x,y,theta,alpha,iterations):
    '''梯度下降'''
    for i in range(iterations):
        inner=x*theta.T
        for j in range(theta.shape[1]):
            theta[0,j]=theta[0,j]-alpha*np.sum(np.multiply(
                        inner-y,x[:,j]))/len(y)
        cost_func(x, y, theta)
    return theta

alpha=0.01
iterations=1000
print(gd(x,y,theta,alpha,iterations))

def theta_transform(theta,means,stds):
    '''求反归一化的theta'''
    theta=np.array(theta.reshape(-1,1))
    means=means.reshape(-1,1)
    stds=stds.reshape(-1,1)
    theta[0]=(theta[0]-np.sum(theta[1:]*means[:-1]/stds[:-1]))*stds[-1]+means[-1]
    theta[1:]=theta[1:]/stds[:-1]*stds[-1]
    return theta

def forecast(theta,my_house):
    price=theta[0]+theta[1]*my_house[0]+theta[2]*my_house[1]
    print(price)

theta=theta_transform(theta,means,stds)
my_house=[2104,3]
forecast(theta,my_house)

fig=plt.figure(figsize=(16,9))
costx=np.linspace(0,1000,1000)
ax=fig.add_subplot(121,projection="3d")
x_3d=np.arange(mins[0],maxs[0]+1,1)
y_3d=np.arange(mins[1],maxs[1]+1,1)
x_3d,y_3d=np.meshgrid(x_3d,y_3d)

z_3d=float(theta[0])+float(theta[1])*x_3d+float(theta[2])*y_3d



ax.view_init(elev=25, azim=125)
ax.set_xlabel("Size")
ax.set_ylabel("Bedroom")
ax.set_zlabel("Price")
ax.plot_surface(x_3d,y_3d,z_3d,rstride=1,cstride=1,color="blue")


ax.scatter(contents1[:,0], contents1[:,1], contents1[:,2],c="r")

fig.add_subplot(122)
plt.plot(costx,costy)  #代价函数收敛曲线
plt.show()