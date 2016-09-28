#include "./Model.h"

int main(int argc, char*argv[])
{
    Model m;
    m.trainSVRModel(argv[1],argv[2]);
    m.testSVRModel(argv[3],argv[2],argv[4]);
    return 0;
}

