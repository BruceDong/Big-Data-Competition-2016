/*****************************************************************************
Copyright:      Xishuang Dong
File name:      Model.h
Description:    Construct various models for forecasting
Author:         Xishuang Dong
Version:        0.1
Date:           2015.05.18
History:        null
*****************************************************************************/
#ifndef _MODEL_H
#define _MODEL_H

#include "./src/svr.h"

class Model{
    private:
    SVR svr;

    public:
    Model();
    ~Model();

    public:
    /***************Machine learning based model*********************/
    bool trainSVRModel(const char * trainFile, const char * modelFile);
    bool testSVRModel(const char * testFile, const char * modelFile, const char * resultFile);

    /***************Machine learning based model*********************/


};

#endif
