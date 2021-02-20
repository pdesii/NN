/*
 * https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547
*/

#include <math.h>
#include "mex.h"

static double sigmoid(double x)
{ 
    return 1 / (1 + exp(-x));
}

static double dSigmoid(double x)
{ 
    return x * (1 - x);
}

static void nnComputeOutput(int numInputs,int numHiddenNodes, int numOutputs, double *inputs,
                double *hiddenLayerBias, double *hiddenWeights, double *outputLayerBias, double *outputWeights,
                double *hiddenLayer, double *outputLayer)
{
    // Compute hidden layer activation
    for (int j=0; j<numHiddenNodes; j++)
    {
        double activation=hiddenLayerBias[j];
        for (int k=0; k<numInputs; k++)
        {
            activation += inputs[k]*hiddenWeights[k + j*numInputs];
        }
        hiddenLayer[j] = sigmoid(activation);
    }
    
    // Compute output layer activation
    for (int j=0; j<numOutputs; j++) 
    {
        double activation=outputLayerBias[j];
        for (int k=0; k<numHiddenNodes; k++) 
        {
            activation += hiddenLayer[k]*outputWeights[k + j*numHiddenNodes];
        }
        outputLayer[j] = sigmoid(activation);
    }
}

static void nnTrain(int numEpochs, double learningRate,
                int numInputs,int numHiddenNodes, int numOutputs, 
                double *training_inputs, double *training_outputs, int numInputTrainingSets,
                double *hiddenLayerBias, double *hiddenWeights, double *outputLayerBias, double *outputWeights,
                double *hiddenLayer, double *outputLayer, double *deltaOutput, double *deltaHidden, double *trainError)
{
    for (int n=0; n < numEpochs; n++) 
    {
        double totError = 0.0;
        unsigned int counter = 0;
        for(int i=0;i<numInputTrainingSets;i++)
        {            
            nnComputeOutput(numInputs, numHiddenNodes, numOutputs, &training_inputs[i*numInputs],
                hiddenLayerBias, hiddenWeights, outputLayerBias, outputWeights,
                hiddenLayer, outputLayer);
            
            // Compute change in output weights
            for (int j=0; j<numOutputs; j++) 
            {
                double dError = (training_outputs[i*numOutputs + j]-outputLayer[j]);
                deltaOutput[j] = dError*dSigmoid(outputLayer[j]);
                
                totError += fabs(dError);
                counter++;
                
                /*mexPrintf("i=%d deltaOutput=%g  %g + %g = %g(%g) \r\n",i, dError,
                training_inputs[i*numInputs],training_inputs[i*numInputs+1],training_outputs[i*numOutputs + j],outputLayer[j]);*/
            }
            
            // Compute change in hidden weights
            for (int j=0; j<numHiddenNodes; j++) 
            {
                double dError = 0.0f;
                for(int k=0; k<numOutputs; k++) 
                {
                    dError+=deltaOutput[k]*outputWeights[j + k*numHiddenNodes];
                }
                deltaHidden[j] = dError*dSigmoid(hiddenLayer[j]);
            }
            
            // Apply change in output weights
            for (int j=0; j<numOutputs; j++) 
            {
                outputLayerBias[j] += deltaOutput[j]*learningRate;
                for (int k=0; k<numHiddenNodes; k++)
                {
                    outputWeights[k + j*numHiddenNodes]+=hiddenLayer[k]*deltaOutput[j]*learningRate;
                }
            }
            // Apply change in hidden weights
            for (int j=0; j<numHiddenNodes; j++) 
            {
                hiddenLayerBias[j] += deltaHidden[j]*learningRate;
                for(int k=0; k<numInputs; k++) 
                {
                    hiddenWeights[k + j*numInputs] +=training_inputs[i*numInputs + k]*deltaHidden[j]*learningRate;
                }
            }
        }
        
        trainError[n] = totError/counter;
//        mexPrintf("Epoch:%d  avgError:%g counter:%d totError:%g\r\n",n,trainError[n],counter,totError);
        mexPrintf("%d  %g \r\n",n,trainError[n]);
    }
}
