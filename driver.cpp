#include "FeedForwardNeuralNetwork.h"

int main(void)
{

         char  * trainfile = "xmackeytrain.txt";// training data set
         char * testfile= "xmackeytest.txt"; //testing data set
         char * saveknowledge = "learntweights.txt";

         ofstream out2;
         out2.open("out2.txt");

         int trainsize = 249;
         int testsize = 249;

         int inputsize = 3;
         int hidden = 3;
         int output = 1;
         double learningrate = 0.1;


	FNNTrainingExamples Samples(trainfile, trainsize,inputsize+output, inputsize,output);  //get data from file 138-lines 16-values_in_line 13-data_input_values 3_output_values  (wine UCI dataset)

	Samples.printData();
       FeedForwardNeuralNetwork network;



		Sizes NNtopology; //vector of network topology [input,hidden,output] - would work with more input layer [input, hidden, hidden, output]. Note "Sizes" is typdef of vector <int>
		//number of neurons in each layer
		NNtopology.push_back(inputsize);//input
		NNtopology.push_back(hidden);//hidden
		NNtopology.push_back(output);//output


		network.BackPropogation(Samples,learningrate,NNtopology, saveknowledge,true);

		double Train =  network.TestTrainingData(NNtopology,saveknowledge, trainsize,trainfile,inputsize,output,out2);
		double Test =  network.TestTrainingData(NNtopology,saveknowledge,testsize,testfile, inputsize,output,out2);

	return 0;
};
