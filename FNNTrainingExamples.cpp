
#include "FNNTrainingExamples.h"


FNNTrainingExamples::FNNTrainingExamples()
{
    //constructor
} ;
//overriden constructor
FNNTrainingExamples::FNNTrainingExamples( char* File, int size, int length, int inputsize, int outputsize ){
    //initialize functions and class variables
    inputcolumnSize = inputsize ;
    outputcolumnSize = outputsize;
    datafilesize = inputsize+ outputsize;
    colSize = length;
    Datapoints= size;

    FileName = File;
    InitialiseData();
}

void FNNTrainingExamples:: InitialiseData()
{
	ifstream in( FileName );
        if(!in) {
		cout << endl << "failed to open file" << endl;//error message for reading from file
        }

	//initialise dataset vectors
	for(int r=0; r <  Datapoints ; r++)
	DataSet.push_back(vector<double> ());

	for(int row = 0; row < Datapoints ; row++) {
		for(int col = 0; col < colSize ; col++)
			DataSet[row].push_back(0);
	}
      // cout<<"printing..."<<endl;
    for(  int row  = 0; row   < Datapoints ; row++)
    for( int col  = 0; col  < colSize; col ++)
      in>>DataSet[row ][col];
      //-------------------------
    //initialise intput vectors
	for(int  r=0; r < Datapoints; r++)
		InputValues.push_back(vector<double> ());

    for(int  row = 0; row < Datapoints ; row++)
		for(int col = 0; col < inputcolumnSize ; col++)
			InputValues[row].push_back(0);//initialise with 0s

    for(int  row = 0; row < Datapoints ; row++)
		for(int col = 0; col < inputcolumnSize ;col++)
			InputValues[row][col] = DataSet[row ][col] ;//read values from the dataset vector

	//initialise output vectors
	for(int r=0; r < Datapoints; r++)
		OutputValues.push_back(vector<double> ());

    for( int row = 0; row < Datapoints ; row++)
		for( int col = 0; col < outputcolumnSize; col++)
			OutputValues[row].push_back(0);//initialse with 0s

    for( int row = 0; row < Datapoints ; row++)
		for(int  col = 0; col <  outputcolumnSize;col++)
			OutputValues[row][col]= DataSet[row ][ col +inputcolumnSize ] ;

    in.close();//close connection
 }

void FNNTrainingExamples:: printData()
{
    cout<<"printing...."<<endl;
	cout<<"Entire Data Set.."<<endl;
	for(int row = 0; row < Datapoints ; row++) {
		for(int col = 0; col < colSize ; col++)
			cout<<  DataSet[row][col]<<" ";//output entire set
			cout<<endl;
	}

	cout<<endl<<"Input Values.."<<endl;

	for(int  row = 0; row < Datapoints ; row++) {
		for( int col = 0; col < inputcolumnSize; col++)
			cout<<  InputValues[row][col]<<" ";//output only input values
			cout<<endl;
    }

	cout<<endl<<"Expected Output Values.."<<endl;

	for( int row = 0; row < Datapoints ; row++)  {
		for( int col = 0; col <  outputcolumnSize;col++)
			cout<< OutputValues[row][col] <<" " ;//print output values
			cout<<endl;
	}
}

