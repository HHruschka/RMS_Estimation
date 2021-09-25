#  Estimation of the Replicated Softmax Model  for  browsing baskets 
#   
#  Slight modification of  the  CD Algorithm for the Replicated Softmax Model 
#  implemented by Daichi Mochihashi, http://chasen.org/~daiti-m/dist/rsm/
#   
#   see the paper aaaaa, bbbbb: xxxxxxxx. J cccccccc (forthcoming )
#   
#  Literature Reference:  
#  Salakhutdinov, R, Hinton, GE (2009) Replicated softmax: An undirected topic model.  In: 
#  Proceedings of the 22nd International Conference on Neural Information Processing Systems, 
#  pp. # 1607–1614


import sys

import rsm_cd
import numpy as np
import csv




# set estimation parameters
nhus  = 3;    # nunber of hidden units 

nstarts = 2    # number of random starts 
epochs = 50    # number of learning epochs
iter = 1       # number of CD iterations
rate = 0.001   # learning constant
batch = 1      # minibatch size 


def sigmoid(X):
    """
    sigmoid of X
    """
    return (1 + np.tanh(X/2))/2

def c_prob(X,W,a,b):
    
   n = np.shape(X)[0]
   XD = X.sum(axis=1)
   h = sigmoid(np.dot(X, W) + np.outer(XD, a))
   # compute visible activations
   v = np.dot(h, W.T) + b
   # exp and normalize.
   tmp = np.exp(v)
   sum = tmp.sum(axis=1)
   sum = sum.reshape((n,1))
   return (tmp / sum)


def main():
 
    global epochs, iter, rate, batch, proto
    
    print ('loading data..'),; sys.stdout.flush()
    
    reader = csv.reader(open('browsing_baskets.csv', "r"), delimiter=",")
    lx = list(reader)
    X = np.array(lx)
  
    print ('done.')
    
    # set  random generator  seed
    np.random.seed(1917)

    
    np.set_printoptions(precision=4, suppress=True) 
 
    X  = X.astype(float) # transform to float
    
   
    for istart in range(0,nstarts):

       print ('number of browsing baskets        = %d' % X.shape[0])
       print ('number of sites         = %d' % X.shape[1])
       print ('number of hidden variables = %d' % nhus)
       print ('number of learning epochs  = %d' % epochs)
       print ('number of CD iterations    = %d' % iter)
       print ('minibatch size             = %d' % batch)
       print ('learning rate              = %g' % rate)
        
       RSM=rsm_cd.RSM()
       (ppl,W,b,a) = RSM.train(X, nhus , epochs, iter, lr=rate, btsz=batch)
          
       print("random start",istart," Perplexity  %.03f" % ppl)
       print("interactions  W_kj");  print(W)
       print("hidden unit constants a_k"); print(a)
       print("site Constants b_j"); print(b)  
           
       prob = c_prob(X,W,a,b) 
     
       print("part of the matrix of visiting probabilities")
       print(prob)
      

 
if __name__ == '__main__':
   main ()

