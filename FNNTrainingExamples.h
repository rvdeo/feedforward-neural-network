#ifndef FNNTRAININGEXAMPLES_H
#define FNNTRAININGEXAMPLES_H

#include "FNNLayers.h"

#include<iostream>
#include<fstream>
#include<stdlib.h>

class FNNTrainingExamples{
    friend class FeedForwardNeuralNetwork;

	protected:
       //class variables
	   Data  InputValues;
       	   Data  DataSet;
           Data  OutputValues;
	   char* FileName;
	   int Datapoints;
           int colSize ;
           int inputcolumnSize ;
           int outputcolumnSize ;
	   int datafilesize  ;

	public:
		//function declarations
		FNNTrainingExamples();
		//overriden constructor
		FNNTrainingExamples( char* File, int size, int length, int inputsize, int outputsize );

		void printData();

		void InitialiseData();

};


#endif // FNNTRAININGEXAMPLES_H
