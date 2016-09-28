#include "svr.h"
#include "svm_learn_main.h"
#include "svm_classify.h"

SVR::SVR()
{
}

SVR::~SVR()
{
}

bool SVR::learn(const char * learnFile, const char * model)
{
    int argcLearn = 7;
    char *lpLearn[] = {const_cast<char*>("svm_learn"),const_cast<char*>("-z"),
                                        const_cast<char*>("r"),const_cast<char*>("-t"),const_cast<char*>("2"),
                                        const_cast<char*>(learnFile),const_cast<char*>(model)};
    svm_learn(argcLearn, lpLearn);

    return true;
}

bool SVR::regression(const char * predictFile, const char * model, const char * result)
{
    int argcPredict = 4;
    char *lpPredict[] = {const_cast<char*>("svm_classify"),const_cast<char*>(predictFile),const_cast<char*>(model),const_cast<char*>(result)};
    svm_regression(argcPredict,lpPredict);
    return true;
}
