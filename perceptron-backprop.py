import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import math

# config MLP: multiplayer perceptron
def createMLP(topology = (2,3,1),eta = 0.25):
    tp = topology
    return {'weights':
                [np.random.random((tp[i]+1,tp[i+1])) for i in range(len(tp)-1)],
            'net_sum':
                [np.zeros(i) for i in tp[1:]],
            'outputs':
                [np.zeros(i) for i in tp[1:]],
            'delta':
                [np.zeros(i) for i in tp[1:]],
            'delta_weights':
                [np.zeros((tp[i] + 1, tp[i + 1])) for i in range(len(tp) - 1)],
            'eta': eta
            }

def perceptron_sigmoid(W,I):
    net =  (I * W).sum()
    return net, 1 / (1 + math.exp(-net)) # logistic function

def evaluate_layer(w_layer,I):
    nets = []
    outs = []
    I_bias = np.append(I, [1])
    for percep in w_layer.T:
        net, o = perceptron_sigmoid(percep,I_bias)
        nets.append(net)
        outs.append(o)
    return np.array(nets), np.array(outs)

def evaluate(mlp,I):
    assert np.shape(mlp['weights'][0])[0] == np.shape(np.append(I, [1]))[0], 'Number of input features and number of weights input layer must be equals'
    mlp['net_sum'][0], mlp['outputs'][0] = evaluate_layer(mlp['weights'][0], I)
    for i, w_layer in enumerate(mlp['weights'][1:]):
        mlp['net_sum'][i+1], mlp['outputs'][i+1] = evaluate_layer(w_layer,mlp['outputs'][i])
    return mlp

def learn(mlp,X,Y):
    rg = range(len(mlp['outputs']))
    rg.reverse()
    # output layer
    mlp['delta'][rg[0]] = mlp['outputs'][rg[0]]*(1 - mlp['outputs'][rg[0]])*(Y - mlp['outputs'][rg[0]])
    Xi = mlp['outputs'][rg[1]] if len(rg) > 1 else X
    np.append(Xi, [1])
    mlp['delta_weight'][rg[0]] = mlp['eta']*mlp['delta'][rg[0]]*Xi
    #hidden layers
    for lidx in rg[1:]:
        mlp['delta'][rg[lidx]] = mlp['outputs'][rg[lidx]] * (1 - mlp['outputs'][rg[lidx]]) * (Y - mlp['outputs'][rg[0]])

    return []

def train(mlp,Xs,Ys):
    for X,Y in zip(Xs,Ys):
        evaluate(mlp,X)
        learn(mlp,X,Y)


mlp = createMLP((2,3,2))
#I = np.array([0,1,0,1,0,0])
#evaluate(mlp,I)