
#include "FeedForwardNeuralNetwork.h"



FeedForwardNeuralNetwork::FeedForwardNeuralNetwork( )
{
    ;//constructor
}

FeedForwardNeuralNetwork::FeedForwardNeuralNetwork(Sizes layer )
{
    layersize  = layer;
    StringSize = (layer[0]*layer[1])+(layer[1]*layer[2]) +   (layer[1] + layer[2]);
}

void FeedForwardNeuralNetwork::PlaceHeuristic(double H)
{
    Heuristic = H;
}

/*
	--Random--
	Is used to generate random numbers which are used to initialize weights and neurons when the network is created
*/
double FeedForwardNeuralNetwork::Random()//method for assigning random weights to Neural Network connections
{
	int chance;
    double randomWeight;
    double NegativeWeight;
    chance =rand()%2;//randomise between positive and negative

    if(chance ==0){
		randomWeight =rand()% 100;
		return randomWeight*0.05;//assign positive weight
    }

    if(chance ==1){
		NegativeWeight =rand()% 100;
		return NegativeWeight*-0.05;//assign negative weight
    }

}
/*
	--Sigmoid--
	Function to convert weighted_sum into a value between  -1 and 1
*/
double FeedForwardNeuralNetwork::Sigmoid(double ForwardOutput)
{

    return  (1.0 / (1.0 + exp(-1.0 * (ForwardOutput) ) ));
}


void FeedForwardNeuralNetwork::CreateNetwork(Sizes Layersize,FNNTrainingExamples TraineeSamples)//create network and initialize the weights
{


	int end = Layersize.size() - 1;

	for(int layer=0; layer < Layersize.size()-1; layer++){//go through each layer

        for(int  r=0; r < Layersize[layer]; r++)
			nLayer[layer].Weights.push_back(vector<double> ());

        for( int row = 0; row< Layersize[layer] ; row++)
			for( int col = 0; col < Layersize[layer+1]; col++)
				nLayer[layer].Weights[row].push_back(Random());

		for(int  r=0; r < Layersize[layer]; r++)
			nLayer[layer].WeightChange.push_back(vector<double> ());

        for( int row = 0; row < Layersize[layer] ; row ++)
			for( int col = 0; col < Layersize[layer+1]; col++)
				nLayer[layer].WeightChange[row ].push_back(0);

        for( int r=0; r < Layersize[layer]; r++)
			nLayer[layer].H.push_back(vector<double> ());//create matrix

        for(int  row = 0; row < Layersize[layer] ; row ++)
			for( int col = 0; col < Layersize[layer+1]; col++)
				nLayer[layer].H[row ].push_back(0);//initialize all the elements in H with 0 for each layer

    }

    for( int layer=0; layer < Layersize.size(); layer++){

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer].Outputlayer.push_back(0);//initialize neurons of each layer with 0s

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].Bias.push_back(Random());//the bias for each each layer and connection will be a random value

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].Gates.push_back(0);//initialize gates vector with 0s


		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].B.push_back(0);//initialize with 0s

		for( int row = 0; row < Layersize[layer] ; row ++)//for each connection we will also keep track of change in bias factor
			nLayer[layer ].BiasChange.push_back(0);//initially it will be all 0

		for( int row = 0; row < Layersize[layer] ; row ++)
			nLayer[layer ].Error.push_back(0);// intialize error vector for each layer with 0s
    }

    for(int  r=0; r < TraineeSamples.Datapoints; r++)
        Output.push_back(vector<double> ());

	for( int row = 0; row< TraineeSamples.Datapoints ; row++)
        for(int  col = 0; col < Layersize[end]; col++)
			Output[row].push_back(0);// intialize all the rows in the output vector with 0s

}


void FeedForwardNeuralNetwork::ForwardPass(FNNTrainingExamples TraineeSamples,int patternNum,Sizes Layersize)
{
   //declaring essential variables
	double WeightedSum = 0;
    double ForwardOutput;//to hold output value between -1 and 1
	int end = Layersize.size() - 1; //know the last layer

    for(int row = 0; row < Layersize[0] ; row ++)
		nLayer[0].Outputlayer[row] = TraineeSamples.InputValues[patternNum][row];


    for(int layer=0; layer < Layersize.size()-1; layer++){
		for(int y = 0; y< Layersize[layer+1]; y++) {
			for(int x = 0; x< Layersize[layer] ; x++){
				WeightedSum += (nLayer[layer].Outputlayer[x] * nLayer[layer].Weights[x][y]);
                 }
				ForwardOutput = WeightedSum - nLayer[layer+1].Bias[y];
			//}

			nLayer[layer+1].Outputlayer[y] = Sigmoid(ForwardOutput);//convert the weighted sum to a value between 1 and -1 which will be the new value in the neuron

			WeightedSum = 0;
		}
		WeightedSum = 0;
	}//end layer

   //--------------------------------------------
   for(int output= 0; output < Layersize[end] ; output ++){
		Output[patternNum][output] = nLayer[end].Outputlayer[output];

	}

 }


void FeedForwardNeuralNetwork::BackwardPass(FNNTrainingExamples TraineeSamp,double LearningRate,int patternNum,Sizes Layersize)
{
	int end = Layersize.size() - 1;// know the end layer
    double temp = 0;

    // compute error gradient for output neurons
	for(int output=0; output < Layersize[end]; output++) {
		nLayer[end].Error[output] = (Output[patternNum][output]*(1-Output[patternNum][output]))*(TraineeSamp.OutputValues[patternNum][output]-Output[patternNum][output]);
    }
    //----------------------------------------

	for(int layer = Layersize.size()-2; layer != 0; layer--){

		for( int x = 0; x< Layersize[layer] ; x++){  //inner layer
			for(int  y = 0; y< Layersize[layer+1]; y++) { //outer layer
				temp += ( nLayer[layer+1].Error[y] * nLayer[layer].Weights[x][y]);
            }
			nLayer[layer].Error[x] = nLayer[layer].Outputlayer[x] * (1-nLayer[layer].Outputlayer[x]) * temp;

			temp = 0.0;//reset temp for the next neuron
		}
		temp = 0.0; //reset temp for the next layer

	}

  	double tmp;
  	//int layer =0;
	for( int layer = Layersize.size()-2; layer != -1; layer--){//go through all layers

		for( int x = 0; x< Layersize[layer] ; x++){  //inner layer
			for( int y = 0; y< Layersize[layer+1]; y++) { //outer layer
				tmp = (( LearningRate * nLayer[layer+1].Error[y] * nLayer[layer].Outputlayer[x])  );
				nLayer[layer].Weights[x][y] += ( tmp  -  ( alpha * tmp) ) ;//update weight

            }
		}
	}

   double tmp1;

    for( int layer = Layersize.size()-1; layer != 0; layer--){//go through all layers

        for( int y = 0; y< Layersize[layer]; y++){
			tmp1 = (( -1 * LearningRate * nLayer[layer].Error[y])  );//calculate change in bias
			nLayer[layer].Bias[y] +=  ( tmp1 - (alpha * tmp1))  ;//updated bias of layer

        }
	}


 }


bool FeedForwardNeuralNetwork::ErrorTolerance(FNNTrainingExamples TraineeSamples,Sizes Layersize, double TrainStopPercent)
{
	//declare essential variables
    double count = 0;
    int total = TraineeSamples.Datapoints;
    double accepted = total;
    double desiredoutput;
    double actualoutput;
    double Error;
    int end = Layersize.size() - 1;

	//go through all training samples
	for(int pattern = 0; pattern< TraineeSamples.Datapoints; pattern++){

		Layer Desired;
		Layer Actual;

		for(int i = 0; i <  Layersize[end] ;i++)
			Desired.push_back(0);//initialize vector for desired output with 0s
		for(int j = 0; j <  Layersize[end] ;j++)
			Actual.push_back(0);//initialize vector for actual output with 0s



		for(int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;

			if((actualoutput >= 0)&&(actualoutput <= 0.2))
				actualoutput = 0;//round down
			else if((actualoutput <= 1)&&(actualoutput >= 0.8))
				actualoutput = 1;//round up

			Actual[output] =  actualoutput;
		}
		int confirm = 0;

		for(int b = 0; b <  Layersize[end] ;b++){
			if(Desired[b]== Actual[b] )
				confirm++;

			if(confirm == Layersize[end])
				count++;
				confirm = 0;//reset for next layer

		}
	}
    if(count ==accepted)
			return false;


	return true;

}


double FeedForwardNeuralNetwork::SumSquaredError(FNNTrainingExamples TraineeSamples,Sizes Layersize)
{   int end = Layersize.size() - 1;//know last layer
    double Sum = 0;
    double Error=0;
    double ErrorSquared = 0;
    for(int pattern = 0; pattern< TraineeSamples.Datapoints ; pattern++){
		for(int output = 0; output < Layersize[end]; output++) {
			Error = fabs(TraineeSamples.OutputValues[pattern][output]) - fabs(Output[pattern][output]);
			ErrorSquared += (Error * Error);//square the error
		}

        Sum += (ErrorSquared);//add to cumulative error
        ErrorSquared = 0;//set error squared variable to 0 for next layer

	}
	return sqrt(Sum/TraineeSamples.Datapoints*Layersize[end]);//return square root of sum / (no. of training samples * no. of neurons in output layer)
}

void FeedForwardNeuralNetwork::PrintWeights(Sizes Layersize)//output the values of all the connection weights
{
    int end = Layersize.size() - 1;

    for(int layer=0; layer < Layersize.size()-1; layer++){

		cout<<layer<<"  Weights::"<<endl<<endl;
		for(int row  = 0; row <Layersize[layer] ; row ++){
			for(int col = 0; col < Layersize[layer+1]; col++)
				cout<<nLayer[layer].Weights[row ][col]<<" "; //output all values from the weights matrix for all layers
				cout<<endl;
        }
		cout<<endl<<layer<<"  WeightsChange::"<<endl<<endl;

		for( int row  = 0; row <Layersize[layer] ; row ++){
			for( int col = 0; col < Layersize[layer+1]; col++)
				cout<<nLayer[layer].WeightChange[row ][col]<<" ";//output all values from the weightchange matrix for all layers
				cout<<endl;
        }

		cout<<"--------------"<<endl;
	}

	for(int layer=0; layer < Layersize.size() ; layer++){
		cout<<endl<<layer<<"  Outputlayer::"<<endl<<endl;//output values from outputlayer
		for( int row = 0; row < Layersize[layer] ; row ++)
			cout<<nLayer[layer].Outputlayer[row] <<" ";

	cout<<endl<<layer<<"  Bias::"<<endl<<endl;
	for( int row = 0; row < Layersize[layer] ; row ++)
        cout<<nLayer[layer].Bias[row] <<" ";//output values from the Bias Matrix for each layer

	cout<<endl<<layer<<"  Error::"<<endl<<endl;
	for(int  row = 0; row < Layersize[layer] ; row ++)
        cout<<nLayer[layer].Error[row] <<" "; //output values from the error matrix for each layer

	cout<<"----------------"<<endl;

}

     }
/*
	--SaveLearnedData--
	Save the network weights and bias values which were able to achieve optimal results to file
*/
void FeedForwardNeuralNetwork::SaveLearnedData(Sizes Layersize, char* filename)//save data to file
{

	ofstream out;
	out.open(filename);
	if(!out) {
		cout << endl << "failed to save file" << endl;//error in writing to file
		return;
    }

    for(int layer=0; layer < Layersize.size()-1; layer++){//ouput weights
        for(int row  = 0; row <Layersize[layer] ; row ++){
			for(int col = 0; col < Layersize[layer+1]; col++)
				out<<nLayer[layer].Weights[row ][col]<<" ";
				out<<endl;
        }
        out<<endl;//blank line
    }

  // output bias.
	for(int  layer=1; layer < Layersize.size(); layer++){
		for(int y = 0 ; y < Layersize[layer]; y++) {
			out<<	nLayer[layer].Bias[y]<<"  ";
			out<<endl<<endl;
        }
	    out<<endl;
    }

	out.close();//data saved--close connection

	return;
}
/*
	--LoadSavedData--
	Load the network weights and bias values which were able to achieve optimal results from file
*/
void FeedForwardNeuralNetwork::LoadSavedData(Sizes Layersize,char* filename)//load saved data from file
{
 	ifstream in(filename);
    if(!in) {
		cout << endl << "failed to save file" << endl;//error reading from file
		return;
    }

	for(int layer=0; layer < Layersize.size()-1; layer++)//read weights
		for(int row  = 0; row <Layersize[layer] ; row ++)
			for(int col = 0; col < Layersize[layer+1]; col++)
				in>>nLayer[layer].Weights[row ][col];


	for( int layer=1; layer < Layersize.size(); layer++)//read bias
		for(int y = 0 ; y < Layersize[layer]; y++)
			in>>	nLayer[layer].Bias[y] ;

	in.close();
	cout << endl << "data loaded for testing" << endl;//data read...close connection
	return;
 }


double FeedForwardNeuralNetwork::CountTestingData(FNNTrainingExamples TraineeSamples,int temp,Sizes Layersize)
{
	//variable declaration
    double count = 0;
    int total = TraineeSamples.Datapoints;
    double accepted =  temp * 1;
    double desiredoutput;
    double actualoutput;
    double Error;
    int end = Layersize.size() - 1;

    for(int pattern = 0; pattern< temp; pattern++){
		//variable declaration
		Layer Desired;//to hold desired output from dataset
		Layer Actual;//to hold actual calculated output

		for(int i = 0; i <  Layersize[end] ;i++)
			Desired.push_back(0);//initiliaze with 0s
		for(int j = 0; j <  Layersize[end] ;j++)
			Actual.push_back(0);//intialize with 0s

		for(int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;
			if((actualoutput >= 0)&&(actualoutput <= 0.5))
				actualoutput = 0;//if its between 0-0.5 then round it down to 0

			else if((actualoutput <= 1)&&(actualoutput >= 0.5))
				actualoutput = 1;//it its between 1 and 0.5 then round it up to 1

			Actual[output] =  actualoutput;//store new actual output value

		}

		int confirm = 0;

		for(int b = 0; b <  Layersize[end] ;b++){
			if(Desired[b]== Actual[b] )//check if the actual and desired output match i.e if the prediction/classification was correct
				confirm++;
        }

		if(confirm == Layersize[end])
			count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output then increase count

		confirm = 0;//reset for next set

	}



  return count;//return count of correctly predicted instances

}


double FeedForwardNeuralNetwork::CountLearningData(FNNTrainingExamples TraineeSamples,int temp,Sizes Layersize)
{
	//variable declaration
    double count = 0;
    int total = TraineeSamples.Datapoints;
    double accepted =  temp * 1;
    double desiredoutput;
    double actualoutput;
    double Error;
    int end = Layersize.size() - 1;

    for(int pattern = 0; pattern< temp; pattern++){

		Layer Desired;//to hold desired output from dataset
		Layer Actual;//to hold calculated output values

		for(int i = 0; i <  Layersize[end] ;i++)
			Desired.push_back(0);//initialize with 0s
		for(int j = 0; j <  Layersize[end] ;j++)
			Actual.push_back(0);//initialize with 0s



		for(int output = 0; output < Layersize[end]; output++) {
			desiredoutput = TraineeSamples.OutputValues[pattern][output];
			actualoutput = Output[pattern][output];

			Desired[output] = desiredoutput;


			if((actualoutput >= 0)&&(actualoutput <= (0+error_tolerance)))
				actualoutput = 0;

			else if((actualoutput <= 1)&&(actualoutput >= (1-error_tolerance)))
				actualoutput = 1;

			Actual[output] =  actualoutput;//set new actual output values

		}

		int confirm = 0;

		for(int b = 0; b <  Layersize[end] ;b++){
			if(Desired[b]== Actual[b] )
				confirm++;//match
        }

		if(confirm == Layersize[end])
			count++;//if an instance is correctly predicted meaning all the values of the output layer neurons match the desired output then increase count

		confirm = 0;

    }



	return count;

}
/*
	--CheckOutput--
	To see if actual and desired output values match
*/
bool FeedForwardNeuralNetwork::CheckOutput(FNNTrainingExamples TraineeSamples,int pattern,Sizes Layersize)
{
	//variable declaration
    int end = Layersize.size() - 1;//know last layer
    double desiredoutput;
    double actualoutput;
	Layer Desired; //to hold desired output from dataset
    Layer Actual; //to hold actual calculated output

    for(int i = 0; i <  Layersize[end] ;i++)
        Desired.push_back(0);//initialize with 0s
    for(int j = 0; j <  Layersize[end] ;j++)
        Actual.push_back(0);//initialize with 0s

    int count = 0;

    for(int output = 0; output < Layersize[end]; output++) {
		desiredoutput = TraineeSamples.OutputValues[pattern][output];
		actualoutput = Output[pattern][output];
		Desired[output] = desiredoutput;
		cout<< "desired : "<<desiredoutput<<"      "<<actualoutput<<endl;


		if((actualoutput >= 0)&&(actualoutput <= 0.5))
			actualoutput = 0;

		else if((actualoutput <= 1)&&(actualoutput >= 0.5))
			actualoutput = 1;

		Actual[output] =  actualoutput;//new actual output value
    }

    cout<<"---------------------"<<endl;

    for(int b = 0; b <  Layersize[end] ;b++){
		if(Desired[b]!= Actual[b] )//if the actual and intended output do not match then return false
			return false;
    }
	return true;
}
/*
	--TestTrainingData--
	Test the trained network with testing data
*/
double FeedForwardNeuralNetwork::TestTrainingData(Sizes Layersize, char* filename,int  size, char* load, int inputsize, int outputsize,ofstream & out2  )
{
    //variable declaration
	bool valid;
    double count = 1;
    int total;
    double accuracy;
	int end = Layersize.size() - 1;

	//load testing data
    FNNTrainingExamples Test(load,size,inputsize+outputsize ,   inputsize,  outputsize );
	//initialize network
    CreateNetwork(Layersize,Test);
	//load saved training data
    LoadSavedData(Layersize,filename);

    for(int pattern = 0;pattern < size ;pattern++){

		ForwardPass(Test,pattern,Layersize);
    }

	for(int pattern = 0; pattern< size; pattern++){
		for(int output = 0; output < Layersize[end]; output++) {

			out2<< Output[pattern][output]   <<" "<<Test.OutputValues[pattern][output]<<" "<<fabs( fabs(Test.OutputValues[pattern][output] )-fabs    (Output[pattern][output]))<<endl;

		}
	}
	out2<<endl;
	out2<<endl;
	//get accuracy of test run
	accuracy = SumSquaredError(Test,Layersize);
	out2<<" RMSE:  " <<accuracy<<endl;
	cout<<"RMSE: " <<accuracy<<" %"<<endl;

	return accuracy;

}
/*
	--TestLearnedData--
	Test the network using the training data
*/
double FeedForwardNeuralNetwork::TestLearnedData(Sizes Layersize, char* filename,int  size, char* load, int inputsize, int outputsize )
{
	//variable declaration
    bool valid;
    double count = 1;
    double total;
    double accuracy;
	//get testing data set
    FNNTrainingExamples Test(filename,size,inputsize+outputsize, inputsize, outputsize);

    total = Test.InputValues.size(); //how many samples to test?
	//initialize network
    CreateNetwork(Layersize,Test);
	//load saved network data
    LoadSavedData(Layersize,load);

    for(int pattern = 0;pattern < total ;pattern++){

		ForwardPass(Test,pattern,Layersize);
    }

	count = CountTestingData(Test,size,Layersize);//get number of correctly predicted instances

	accuracy = (count/total)* 100;//get accuracy percentage
	cout<<"The sucessful count is "<<count<<" out of "<<total<<endl;
	cout<<"The accuracy of test is: " <<accuracy<<" %"<<endl;
	//return accuracy %
	return accuracy;

}

int FeedForwardNeuralNetwork::BackPropogation(  FNNTrainingExamples TraineeSamples, double LearningRate,Sizes Layersize, char * Savefile, bool load)
{
    //variable declaration
	double SumErrorSquared;
	int Id = 0;
	int Epoch = 0;
	bool Learn = true;


    CreateNetwork(Layersize,TraineeSamples );//structure the network and initialize the weights & neurons
    cout<< " xx " <<endl;


    while( Learn == true){//learning from training data

		for(int pattern = 0; pattern < TraineeSamples.InputValues.size(); pattern++)//keep doing for all training values
		{
			ForwardPass( TraineeSamples,pattern,Layersize);//pass through the network and check output

			BackwardPass(TraineeSamples,LearningRate,pattern,Layersize);//check back for error

		}


		Epoch++;//number of iterations
		cout<<Epoch<< " : is Epoch    *********************    "<<endl;//output iteration no.

		SumErrorSquared = SumSquaredError(TraineeSamples,Layersize);//calculate sum squared error
		cout<<SumErrorSquared<< " : is SumErrorSquared"<<endl;//show error--error should gradually decrease



		double count = CountLearningData(TraineeSamples,TraineeSamples.InputValues.size(), Layersize);//get count of correctly predicted instances
		SaveLearnedData(Layersize, Savefile);//save weights and network structure to file

		//double trained= count/TraineeSamples.InputValues.size()*100; //get percentage of correctly predicted instances
		//cout<<trained<<" is percentage trained"<<endl;//ouput training percentage





		/*if(trained>=MinTrain){//if accuracy is greater than or equal to 95% then stop learning
			Learn = false;
        }
*/

		if(Epoch ==  MaxTime  ){
			Learn = false;
		}

	}
	return Epoch;//return no. of iterations
}
