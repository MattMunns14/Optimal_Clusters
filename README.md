# Optimal_Clusters
Techniques for efficiently finding optimal number of clusters.

Right now, the script Bayesian optimization and brute force search to find an optimal number of clusters to use with a clustering algorithm. Silhoutte score is used to evaluate how well *k* clusters fits the data. 
There is also a class for generating random two dimensional normal data to test the efficiency and accuracy of the optimization methods. 

I'd like to add additional clustering methods (GMM with EM, Bayesian GMM) and try some other optimization/searching methods. I'd also like to improve the test data generation to generate data with a variable number of dimensions. 
