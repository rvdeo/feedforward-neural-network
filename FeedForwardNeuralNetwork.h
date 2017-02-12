#ifndef FEEDFORWARDNEURALNETWORK_H
#define FEEDFORWARDNEURALNETWORK_H

#include "FNNTrainingExamples.h"
#include<math.h>


const int LayersNumber = 3; //total number of layers.  Can be more if you wish to have mroe than one hidden layer. Need to change topology in main if increase more than 1 hidden layer. i.e layer is 4, topology would be [input, hidden, hidden, output]
const double alpha = 0.00001; //weight decay constant
const double error_tolerance=0.2;//amount of error accepted during training
const int MaxTime  = 2000; // Epochs
const double MinTrain = 95.00; //stop if this value is reached in training performance


class FeedForwardNeuralNetwork{

    protected:
		//class variables
		FNNLayers nLayer[LayersNumber];
		double Heuristic;
		int StringSize;
		Layer ChromeNeuron;
		int NumEval;
		Data Output;
                Sizes layersize;
		double NMSE;

     public:
		//function declaration
		FeedForwardNeuralNetwork( );


		FeedForwardNeuralNetwork(Sizes layer );

		void PlaceHeuristic(double H);

	        //Some functions   are not currently used. You can use them depending on your requirements.
		double Random();

		double Sigmoid(double ForwardOutput);

		double NMSError() {return NMSE;} // not used - good for time series problems


		void CreateNetwork(Sizes Layersize ,FNNTrainingExamples TraineeSamples);

		void ForwardPass(FNNTrainingExamples TraineeSamp,int patternNum,Sizes Layersize);

		void BackwardPass(FNNTrainingExamples TraineeSamp,double LearningRate,int patternNum,Sizes Layersize);

		void PrintWeights(Sizes Layersize);// print  all weights

		bool ErrorTolerance(FNNTrainingExamples TraineeSamples,Sizes Layersize, double TrainStopPercent);


		double SumSquaredError(FNNTrainingExamples TraineeSamples,Sizes Layersize);

		int BackPropogation(  FNNTrainingExamples TraineeSamples, double LearningRate,Sizes Layersize,char* Savefile,bool load);

		void SaveLearnedData(Sizes Layersize,char* filename) ;


		void LoadSavedData(Sizes Layersize,char* filename) ;

		double TestLearnedData(Sizes Layersize,char* filename,int  size, char* load,  int inputsize, int outputsize );

		double  CountLearningData(FNNTrainingExamples TraineeSamples,int temp,Sizes Layersize);

        	double  TestTrainingData(Sizes Layersize, char* filename,int  size, char* load,  int inputsize, int outputsize , ofstream & out2 );

		double CountTestingData(FNNTrainingExamples TraineeSamples,int temp,Sizes Layersize);

		bool CheckOutput(FNNTrainingExamples TraineeSamples,int pattern,Sizes Layersize);
};


#endif // FEEDFORWARDNEURALNETWORK_H
