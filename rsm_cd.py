


import numpy as np

class RSM(object):
    def train(self, data, units, epochs=1000, iter=1, lr=0.001, weightinit=0.001, 
            momentum=0.9, btsz=1):
        """
        CD-n training of RSM.
        @param data: a (rowwise) sample matrix. Number of samples should be divisible by btsz.
        @param units: #latent topics
        @param epochs: #training epochs
        @param lr: learning rate
        @param weightinit: scaling of random weight initialization
        @param momentum: momentum rate
        @param btsz: batchsize   
        """
        dictsize = data.shape[1]
        # initialize weights
        # np.random.seed(1953)   seed may be  fixed to get the same random values
        w_vh = weightinit * np.random.randn(dictsize, units)
        w_v = weightinit * np.random.randn(dictsize)
        w_h = np.zeros((units))
        # weight updates
        wu_vh = np.zeros((dictsize, units))
        wu_v = np.zeros((dictsize))
        wu_h = np.zeros((units))
        delta = lr/btsz
        batches = data.shape[0]//btsz
        print ("updates per epoch: %s | total updates: %s" %(batches, batches*epochs))
        words = np.sum(data)
        for epoch in range(epochs):
            lik = 0
            # visit data randomly
            np.random.shuffle(data)
            # gradually increase iter
            for b in range(batches):
                start = b * btsz 
                v1 = data[start : start+btsz]
                # hidden biases scaling factor
                D = v1.sum(axis=1)
                # project into hidden
                h1 = sigmoid((np.dot(v1, w_vh) + np.outer(D, w_h)))
                v2 = v1; h2 = h1
                for i in range(iter):
                    (v2,h2,z) = cdn (v2,h2,w_vh,w_v,w_h,D)
                    if i == 0:
                        lik += z
                # compute updates
                wu_vh = wu_vh * momentum + np.dot(v1.T, h1) - np.dot(v2.T, h2)
                wu_v = wu_v * momentum + v1.sum(axis=0) - v2.sum(axis=0)
                wu_h = wu_h * momentum + h1.sum(axis=0) - h2.sum(axis=0)
                # update 
                w_vh += wu_vh * delta 
                w_v += wu_v * delta
                w_h += wu_h * delta
            ppl = np.exp (- lik / words)
          
            print ("Epoch[%2d] :  Perplexity = %.02f"  % (epoch,ppl))
     
        return ppl,w_vh,w_v,w_h,
               

def cdn (v1,h1,w_vh,w_v,w_h,D):
    """
    one-step contrastive divergence: (v1,h1)->(v2,h2).
    """
    lik = 0
    btsz = v1.shape[0]
    # project into visible
    v2 = np.dot(h1, w_vh.T) + w_v
    tmp = np.exp(v2)
    sum = tmp.sum(axis=1)
    sum = sum.reshape((btsz,1))
    v2pdf = tmp / sum
    # perplexity
    lik += np.nansum(v1 * np.log(v2pdf))
    # sample from multinomial
    v2 *= 0
    for i in range(btsz):
        v2[i] = np.random.multinomial(D[i],v2pdf[i],size=1)
    # project into hidden
    h2 = sigmoid(np.dot(v2, w_vh) + np.outer(D, w_h))
    return (v2,h2,lik)

def sigmoid(X):
    """
    sigmoid of X
    """
    return (1 + np.tanh(X/2))/2
