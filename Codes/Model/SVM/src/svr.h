#ifndef _SVR_
#define _SVR

class SVR{
public:
    SVR();
    ~SVR();

public:
    bool learn(const char * learnFile, const char * model);
    bool regression(const char * predictFile, const char * model, const char * result);

};

#endif
