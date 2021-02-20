#include "nn.c"

#include "mex.h"
void mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5)
    {
        mexErrMsgTxt ("usage: [newNN trainError] = nnTrain(NN, numEpochs, learningRate, trainingInputs, trainingOutputs)");
        return;
    }
    
    double *nn = mxGetPr(prhs[0]);
    
    int numInputs = *nn++;
    int numHiddenNodes = *nn++;
    int numOutputs = *nn++;
    mexPrintf("numInputs = %d\n",numInputs);
    mexPrintf("numHiddenNodes = %d\n",numHiddenNodes);
    mexPrintf("numOutputs = %d\n",numOutputs);
    
    int numEpochs = mxGetScalar(prhs[1]);
    mexPrintf("numEpochs: %d\n",numEpochs);

    double learningRate = mxGetScalar(prhs[2]);
    mexPrintf("learningRate: %g\n",learningRate);

    //INPUTS
    const mwSize *dimensions = mxGetDimensions(prhs[3]);    
    int nInputs = dimensions[0];
    int numInputTrainingSets = dimensions[1];
    double *training_inputs = mxGetPr(prhs[3]);

    mexPrintf("nInputs = %d   numInputTrainingSets = %d\n",nInputs,numInputTrainingSets);

    if( nInputs != numInputs)
    {
        mexPrintf("error: nInputs(%d) != numInputs(%d)\r\n",nInputs,numInputs);
        return;
    }

    //OUTPUTS
    dimensions = mxGetDimensions(prhs[4]);    
    int nOutputs = dimensions[0];
    int numOutputTrainingSets = dimensions[1];
    double *training_outputs = mxGetPr(prhs[4]);

    mexPrintf("nOutputs = %d   numOutputTrainingSets = %d\n",nOutputs,numOutputTrainingSets);

    if( nOutputs != numOutputs)
    {
        mexPrintf("error: nOutputs(%d) != numOutputs(%d)\r\n",nOutputs,numOutputs);
        return;
    }
    
    if(numOutputTrainingSets != numInputTrainingSets)
    {
        mexPrintf("error: numOutputTrainingSets(%d) != numInputTrainingSets(%d)\r\n",numOutputTrainingSets,numInputTrainingSets);
        return;
    }        

    //New NN
    int size =    1                         //double numInputs
                + 1                         //double numHiddenNodes
                + 1                         //double numOutputs
                + numHiddenNodes            //double hiddenLayerBias[numHiddenNodes]
                + numInputs*numHiddenNodes  //double hiddenWeights[numInputs][numHiddenNodes]
                + numOutputs                //double outputLayerBias[numOutputs]
                + numHiddenNodes*numOutputs //double outputWeights[numHiddenNodes][numOutputs]
                + numHiddenNodes            //double hiddenLayer[numHiddenNodes];
                + numOutputs                //double outputLayer[numOutputs]; 
                + numOutputs                //double deltaOutput[numOutputs];
                + numHiddenNodes;           //double deltaHidden[numHiddenNodes];
    
    plhs[0] = mxCreateDoubleMatrix(size, 1, mxREAL);
    
    nn = mxGetPr(prhs[0]);
    double *nnNew = mxGetPr(plhs[0]);
    
    //Copio la nn in nnNew
    for (int i=0;i<size;i++)
    {
        nnNew[i] = nn[i];
    }
    
    nnNew += 3;
    
    double *hiddenLayerBias = nnNew;
    nnNew += numHiddenNodes;
    
    double *hiddenWeights = nnNew;
    nnNew += numInputs*numHiddenNodes;
    
    double *outputLayerBias = nnNew;
    nnNew += numOutputs;
    
    double *outputWeights = nnNew;
    nnNew += numHiddenNodes*numOutputs;
    
    double *hiddenLayer = nnNew;
    nnNew += numHiddenNodes;

    double *outputLayer = nnNew;
    nnNew += numOutputs;
    
    double *deltaOutput = nnNew;
    nnNew += numOutputs;
    
    double *deltaHidden = nnNew;
    nnNew += numHiddenNodes;
    
    
     
    plhs[1] = mxCreateDoubleMatrix(numEpochs, 1, mxREAL);
    double *trainError = mxGetPr(plhs[1]);
    
    //... and now train it!!
    nnTrain(numEpochs, learningRate,
                numInputs, numHiddenNodes, numOutputs, 
                training_inputs, training_outputs, numInputTrainingSets,
                hiddenLayerBias, hiddenWeights, outputLayerBias, outputWeights,
                hiddenLayer, outputLayer, deltaOutput, deltaHidden, trainError);
}
