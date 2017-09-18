import numpy as np
import matplotlib.pyplot as plt
import matplotlib

h_error = []
ep_error = []

def perceptron(W,I):
    I_bias = np.append(I,[1])
    sm =  (I_bias * W).sum()
    return 1 if sm > 0 else -1

def error(true_y,y):
    return true_y - y

def learn(W,Xi,t,o,eta=0.02):
    X_i = np.append(Xi, [1])
    e = error(t,o)
    h_error.append(e)
    deltaW = eta*e*X_i
    W = W + deltaW
    return W, e

def train(W,X,Y):
    ep_err = 0
    for i, Xi in enumerate(X):
        o = perceptron(W,Xi)
        W, err = learn(W,Xi,Y[i],o)
    ep_err += abs(err)
    return W, ep_err / len(X)

def evaluate_train(W,X,Y,epochs=200):
    for ep in range(0,epochs):
        W, err = train(W,X,Y)
        ep_error.append(err)
    return W

def evaluate(W,X):
    for i, Xi in enumerate(X):
        o = perceptron(W,Xi)
        print "Evaluating: ", Xi, "Output: ", o


Weights = np.random.random(3)
data = np.array([[1,0,1],[0,0,-1],[0,1,1],[1,1,-1]])
all_X = data[:,0:2]
all_Y = data[:,2]

Weights = evaluate_train(Weights,all_X,all_Y,epochs=50)
evaluate(Weights,all_X)

fig, (axis1, axis2, axis3) = plt.subplots(1,3)

axis1.plot(np.abs(np.array(h_error)))
axis1.set_xlabel("Presentation")
axis1.set_ylabel("Error")

print ep_error
axis2.plot(np.array(ep_error))
axis2.set_xlabel("Epoch")
axis2.set_ylabel("Error")

colors = ["black","yellow"]
axis3.scatter(all_X[:,0],all_X[:,1],c=all_Y,cmap=matplotlib.colors.ListedColormap(colors))
axis3.set_xlabel("x0")
axis3.set_ylabel("x1")

canonical = np.array([[0,-Weights[2]/Weights[1]],[-Weights[2]/Weights[0],0]])
axis3.plot(canonical[:,0],canonical[:,1],linestyle="dashed")

plt.show()