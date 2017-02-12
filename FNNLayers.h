#ifndef FNNLAYERS_H
#define FNNLAYERS_H

#include <vector>

using namespace std;

//Type declaratiobns
typedef vector<double> Layer;//for the different layers within the neural network
typedef vector<double> Nodes;//for the nodes within the network-- a vector matrix of type double.
typedef vector<int> Sizes; // Vector of Int
typedef vector<double> Frame;
typedef vector<vector<double> > Weight;//for managing the weights of the neural network connections-=-a vector matrix of type double
typedef vector<vector<double> > Data;


class FNNLayers{

    friend class FeedForwardNeuralNetwork;

    protected:
		//class variables
        Weight Weights ;//neural network weights
        Weight WeightChange;//change in weight after each iteration
        Weight H;
        Layer Outputlayer;
        Layer Bias;//keeping track of the bias factor in the network
        Layer B;
        Layer Gates;
        Layer BiasChange;//keeping track of change in Bias factor
        Layer Error;//error after each iteration

    public:
		//functions
		FNNLayers();

};


#endif // FNNLAYERS_H
