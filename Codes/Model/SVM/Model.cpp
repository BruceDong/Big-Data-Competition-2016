#include "./Model.h"

Model::Model()
{}

Model::~Model()
{}

/*************************************************
Function:       trainSVRModel
Description:    Training SVR model with training data
Input:          trainFile:  training data
Output:         modelFile:  model file
Return:         true for succeed; false for failed
Others:         null
*************************************************/
bool Model::trainSVRModel(const char * trainFile, const char * modelFile)
{
    if(!svr.learn(trainFile, modelFile))
    {
        return false;
    }
    return true;
}


/*************************************************
Function:       testSVRModel
Description:    Testing SVR model with testing data
Input:          testFile:   testing data
                modelFile:  model file
Output:         resultFile: testing results
Return:         true for succeed; false for failed
Others:         null
*************************************************/
bool Model::testSVRModel(const char * testFile, const char * modelFile, const char * resultFile)
{
    if(!svr.regression(testFile, modelFile, resultFile))
    {
        return false;
    }
    return true;
}

