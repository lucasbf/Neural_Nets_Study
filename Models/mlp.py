import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import math

# config MLP: multiplayer perceptron
def createMLP(topology = (2,3,1),activation = 'tanh', eta = 0.05, alpha = 0.1):
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
            'eta': eta,
            'alpha': alpha,
            'activation': tanh if activation == 'tanh' else logistic
            }

def logistic(X,derivative = False):
    return 1 / (1 + math.exp(-X)) if derivative == False else X*(1-X)

def tanh(X,derivative = False):
    return np.tanh(X) if derivative == False else 1.0 - X**2

def perceptron_sigmoid(W,I,activation):
    net =  np.dot(I,W)
    return net, activation(net) # logistic function

def evaluate_layer(w_layer,I,activation):
    nets = []
    outs = []
    I_bias = np.append(I, [1])
    for percep in w_layer.T:
        net, o = perceptron_sigmoid(percep,I_bias,activation)
        nets.append(net)
        outs.append(o)
    return np.array(nets), np.array(outs)

def evaluate(mlp,I):
    assert np.shape(mlp['weights'][0])[0] == np.shape(np.append(I, [1]))[0], 'Number of input features and number of weights input layer must be equals'
    mlp['net_sum'][0], mlp['outputs'][0] = evaluate_layer(mlp['weights'][0], I, mlp['activation'])
    for i, w_layer in enumerate(mlp['weights'][1:]):
        mlp['net_sum'][i+1], mlp['outputs'][i+1] = evaluate_layer(w_layer,mlp['outputs'][i],mlp['activation'])
    return mlp

def evaluate_set(mlp,Xs):
    outputs = []
    for X in Xs:
        evaluate(mlp,X)
        outputs.append(mlp['outputs'][len(mlp['outputs'])-1])
    return np.array(outputs)

def sum_delta_weight(weights,delta):
    sdw = []
    for w in weights[:-1]:
        sdw.append(np.dot(w,delta))
    return np.array(sdw)

def delta_weight(eta,delta,Xi):
    Xi_bias = np.append(Xi, [1])
    dw = np.zeros([Xi_bias.shape[0],delta.shape[0]])
    for i, dlt in enumerate(delta):
        dw[:,i] = eta*dlt*Xi_bias
    return np.reshape(dw,(-1,1)) if len(dw.shape) == 1 else dw

def learn(mlp,X,Y):
    weight_n_1 = []
    for w in mlp['weights']:
        weight_n_1.append(np.copy(w))

    outlidx = len(mlp['outputs'])-1
    # output layer
    mlp['delta'][outlidx] = mlp['activation'](mlp['outputs'][outlidx],derivative=True)*(Y - mlp['outputs'][outlidx])
    Xi = mlp['outputs'][outlidx-1] if outlidx > 0 else X
    mlp['delta_weights'][outlidx] = delta_weight(mlp['eta'],mlp['delta'][outlidx],Xi)

    #hidden layers
    for lidx in range(outlidx-1,-1,-1):
        sdw = sum_delta_weight(mlp['weights'][lidx+1],mlp['delta'][lidx+1])
        mlp['delta'][lidx] = mlp['activation'](mlp['outputs'][lidx],derivative=True) * sdw
        Xi = X if lidx == 0 else mlp['outputs'][lidx-1]
        mlp['delta_weights'][lidx] = delta_weight(mlp['eta'],mlp['delta'][lidx],Xi)

    #update weights
    for weight,delta_w,w_n_1 in zip(mlp['weights'],mlp['delta_weights'],weight_n_1):
        weight += delta_w + mlp['alpha']*w_n_1

def train(mlp,Xs,Ys,epochs=100):
    err_ep = np.zeros(epochs)
    for ep in range(epochs):
        err = 0.0
        for X,Y in zip(Xs,Ys):
            evaluate(mlp,X)
            learn(mlp,X,Y)
            err += abs(mlp['delta'][-1])
        err_ep[ep] = err / X.shape[0]
        if ep % 1000 == 0:
            print "Epoch: ", ep
    return err_ep

def benchmark_xor_mlp(experiment = 'eta',epochs = 3000):
    X = np.array([[1,0],[0,1],[0,0],[1,1]])
    Y = np.array([1,1,0,0])
    if experiment == 'simple':
        fig, axis = plt.subplots(1,1)
        fig.tight_layout()
        mlp = createMLP((2, 2, 1), alpha=0.0, activation='tanh')
        error_training = train(mlp, X, Y, epochs=epochs)
        print error_training
        print zip(X, Y, evaluate_set(mlp, X))
        axis.plot(np.array(error_training))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Error")
    elif experiment == 'simpleN':
        fig, axis = plt.subplots(1,1)
        fig.tight_layout()
        for i in range(10):
            mlp = createMLP((2, 2, 1), alpha=0.0, activation='tanh')
            error_training = train(mlp, X, Y, epochs=epochs)
            print error_training
            print zip(X, Y, evaluate_set(mlp, X))
            axis.plot(np.array(error_training))
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Error")
    elif experiment == 'eta':
        etas = [0.01,0.025,0.05,0.075,0.1,0.2,0.4,0.8,1.0,1.5,2.0,3.0]
        d = len(etas)/4
        fig, axis = plt.subplots(4,d)
        fig.tight_layout()
        for j, eta in enumerate(etas):
            for i in range(5):
                mlp = createMLP((2,2,1),alpha=0.0,activation='tanh',eta=eta)
                error_training = train(mlp,X,Y,epochs=epochs)
                print error_training
                print zip(X, Y, evaluate_set(mlp,X))
                axis[j/d,j%d].plot(np.array(error_training))
                axis[j/d,j%d].set_xlabel("Epoch")
                axis[j/d,j%d].set_ylabel("Error")
                axis[j/d,j%d].set_title("Eta: "+str(eta))
    plt.show()