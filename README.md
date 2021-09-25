# RMS_Estimation

This Python code is a slight modification of the CD Algorithm for the Replicated Softmax Model implemented by Daichi Mochihashi, http://chasen.org/~daiti-m/dist/rsm/. The application of RSM to browsing baskets is explained in Falke, Hruschka: Analyzing Browsing Across Websites by Machine Learning Methods, Journal of Business Research (under review). 

Estimation parameters consist of the number of hidden units (=hidden variables), the number of random starts, the number of learning epochs, the number of CD iterations, the batch size , and the learning constant. These parameters are set in rsm.py and may be changed there if deemed appropriate. 

For information on these parameters and estimation of the RSM in general you may consult Salakhutdinov, R, Hinton, GE (2009) Replicated softmax: An undirected topic model. In: Proceedings of the 22nd International Conference on Neural Information Processing Systems, pp. 1607–1614.

The program reads a csv formatted file (browsing_baskets.csv), which contains an example data set of 200 browsing baskets. For each basket visit frequencies to each of 20 websites are given. For reasons of anonymity websites' names are not included.  The program outputs the perplexity of the estimated model, its coefficients, and visiting probabilities based on these coefficients. 
