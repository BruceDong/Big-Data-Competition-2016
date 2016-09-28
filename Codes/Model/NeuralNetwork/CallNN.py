__author__ = 'Xishuang Dong'

import sys
import getopt
import subprocess

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

__doc__ = 'Call Linear Regression. ' \
          '> python CallNN -t trainfile -e testfile -o result'

def call_lr(train_path,test_path,result):
    command = 'Rscript'

    path2script = 'NN.r' 

    args = [train_path,test_path,result] 
    cmd = [command, path2script] + args 
    subproc = subprocess.check_output(cmd, universal_newlines=True)

def main(argv=None):
    trainingDataFile = ""
    testingDataFile = ""  
    outputFile = ""
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht:e:o:", ["help","input=","output="])
        except getopt.error, msg:
             raise Usage(msg)

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print __doc__
                sys.exit(0)
            elif opt in ("-t", "--training"):
                trainingDataFile = arg
            elif opt in ("-e", "--testing"):
                testingDataFile = arg
            elif opt in ("-o", "--output"):
                outputFile = arg

        if(trainingDataFile == "" or testingDataFile == "" or outputFile ==""):
            print __doc__
            sys.exit(0)

        call_lr(trainingDataFile, testingDataFile, outputFile)


    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    main()

