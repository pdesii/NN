#include "nn.c"

#include "mex.h"
void mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
    {
        mexErrMsgTxt ("usage: out = asrc(NN, input)");
        return;
    }
    
    double *nn = mxGetPr(prhs[0]);
    
    int numInputs = *nn++;
    int numHiddenNodes = *nn++;
    int numOutputs = *nn++;
    
    mexPrintf("numInputs = %d\n",numInputs);
    mexPrintf("numHiddenNodes = %d\n",numHiddenNodes);
    mexPrintf("numOutputs = %d\n",numOutputs);
    
    double *hiddenLayerBias = nn;
    nn += numHiddenNodes;
    
    double *hiddenWeights = nn;
    nn += numInputs*numHiddenNodes;
    
    double *outputLayerBias = nn;
    nn += numOutputs;
    
    double *outputWeights = nn;
    nn += numHiddenNodes*numOutputs;

    double *hiddenLayer = nn;
    nn += numHiddenNodes;

    double *outputLayer = nn;
    nn += numOutputs;

    const mwSize *dimensions = mxGetDimensions(prhs[1]);    
    int nInputs = dimensions[0];
    int numInputSets = dimensions[1];
    double *inputs = mxGetPr(prhs[1]);

    mexPrintf("nInputs = %d   numInputSets = %d\n",nInputs,numInputSets);

    if( nInputs != numInputs)
    {
        mexPrintf("error: nInputs(%d) != numInputs(%d)\r\n",nInputs,numInputs);
        return;
    }

    plhs[0] = mxCreateDoubleMatrix(numOutputs, numInputSets, mxREAL);
    /* get a pointer to the real data in the output matrix */
    double *outputs = mxGetPr(plhs[0]);

    for(int i=0;i<numInputSets;i++)
    {
        nnComputeOutput(numInputs, numHiddenNodes, numOutputs, &inputs[i*numInputs],
                hiddenLayerBias, hiddenWeights, outputLayerBias, outputWeights,
                hiddenLayer, outputLayer);
        for(int j=0;j<numOutputs;j++)
        {
            outputs[i*numOutputs + j] = outputLayer[j];
        }
    }
}

