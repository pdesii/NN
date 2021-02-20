#include "mex.h"
void mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3)
    {
        mexErrMsgTxt ("usage: NN = nnCreate(numInputs, numHiddenNodes, numOutputs)");
        return;
    }
    int numInputs = mxGetScalar(prhs[0]);
    mexPrintf("numInputs = %d\n",numInputs);

    int numHiddenNodes = mxGetScalar(prhs[1]);
    mexPrintf("numHiddenNodes = %d\n",numHiddenNodes);

    int numOutputs = mxGetScalar(prhs[2]);
    mexPrintf("numOutputs = %d\n",numOutputs);
    
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
    
    double *nn = mxGetPr(plhs[0]);
    
    nn[0] = numInputs;
    nn[1] = numHiddenNodes;
    nn[2] = numOutputs;
    
    //Random initializazion for hiddenLayerBias and hiddenWeights 
    for(int i=0;i<numInputs*numHiddenNodes + numHiddenNodes;i++)
        nn[3 + i] = ((double)rand())/((double)RAND_MAX);
    
    //Random initializazion for outputLayerBias + outputWeights
    for(int i=0;i<numHiddenNodes*numOutputs+numOutputs;i++)
        nn[3+numHiddenNodes+numInputs*numHiddenNodes + i] = ((double)rand())/((double)RAND_MAX);
    
    mexPrintf("size = %d\n",size);
}
