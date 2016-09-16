__author__ = 'Xishuang Dong'
import sys
import read_classification_data as rd
import getopt
import os.path
import math
import tensorflow as tf
import numpy as np
import progressbar as pb
import math
import pandas as pd
__doc__ = 'Conduct CNN Classification to classification. ' \
          '> python CNN_Classification.py -t ./../../Data/Training.txt -l ./../../Data/TrainingLabel.txt ' \
          '-m model -e ./../../Data/Testing.txt -p ./../../Data/TestingLabel.txt -o result'
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
class DeepLearning:
    trainingDataFile = ""
    trainingLabelFile = ""
    modelFile = ""
    testingDataFile = ""
    testingLabelFile = ""
    outputFile = ""
    numVolumes = 1
    numPoints = 7
    iterationTimes = 1000
    randomSamplingNumberForTraining = 50
    batchNumberForTesting = 1000
    x = []
    y_ = []
    W_conv1 = []
    b_conv1 = []
    x_image = []
    h_conv1 = []
    h_pool1 = []
    W_conv2 = []
    b_conv2 =[]
    h_conv2 = []
    h_pool2 = []
    W_fc1 = []
    b_fc1 = []
    h_pool3_flat =[]
    h_fc1 =[]
    keep_prob = []
    h_fc1_drop = []
    W_fc2 = []
    b_fc2 = []
    saver = []
    def __init__( self, trainingDataFile,trainingLabelFile, modelFile,testingDataFile,testingLabelFile, outputFile):
        self.trainingDataFile = trainingDataFile
        self.trainingLabelFile = trainingLabelFile
        self.modelFile = modelFile
        self.testingDataFile = testingDataFile
        self.testingLabelFile = testingLabelFile
        self.outputFile = outputFile
    @classmethod
    def weight_variable(cls, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
    @classmethod
    def bias_variable(cls, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)
    @classmethod
    def conv2d(cls,x,W):
        return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
    @classmethod
    def max_pool_2_2(cls, x):
       return tf.nn.max_pool(x, ksize = [1,1,2,1], strides = [1,1,2,1], padding = 'SAME')
    @classmethod
    def init_neural_network(self):
        numFeatures = self.numPoints*self.numVolumes
        self.x = tf.placeholder("float", [None, numFeatures])
        self.y_ = tf.placeholder("float", [None, 2])
        self.W_conv1 = self.weight_variable([1,3,1,16])
        self.b_conv1 = self.bias_variable([16])
        self.x_image = tf.reshape(self.x, [-1,self.numVolumes,self.numPoints,1])
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2_2(self.h_conv1)
        self.W_conv2 = self.weight_variable([1,3,16,32])
        self.b_conv2 = self.bias_variable([32])
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2_2(self.h_conv2)
        self.W_fc1 = self.weight_variable([int(math.ceil(self.numVolumes/4.0))*int(math.ceil(self.numPoints/4.0))*32, 1024])
        self.b_fc1 = self.bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, int(math.ceil(self.numVolumes/4.0))*int(math.ceil(self.numPoints/4.0)*32)])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.W_fc2 = self.weight_variable([1024,2])
        self.b_fc2 = self.bias_variable([2])
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)
        self.saver = tf.train.Saver()
    def start_train(self):
        numFeatures = self.numPoints*self.numVolumes;
        print ("Reading training data...")
        train_sample = rd.read_data(self.trainingDataFile)
        train_label  = rd.read_data(self.trainingLabelFile)
        sampleNumber = len(train_label)
        print 'Sample Number is ', sampleNumber
        print ("Training model...")
        sess = tf.InteractiveSession()
        print ("Initializing Neural Network...")
        self.init_neural_network()
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        sess.run(init)
        for i in range(self.iterationTimes):
       		batch_xs, batch_ys = rd.random_read_next_batch(train_sample, train_label,sampleNumber, self.randomSamplingNumberForTraining,numFeatures)
         	if i%100 == 0:
          		train_accuracy = accuracy.eval(feed_dict = {self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
          		print "step %d, training accuracy %g"%(i, train_accuracy)
         	train_step.run(feed_dict = {self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
        if not os.path.isdir("./Model"):
            	os.makedirs("./Model")
        model = os.path.join('./Model', self.modelFile)
        save_path = self.saver.save(sess, model)
        print 'Model saved in file: ', save_path
        sess.close()
    def start_test(self):
        #self.init_neural_network()
      	numFeatures = self.numPoints*self.numVolumes
        sess = tf.InteractiveSession()
        print ('Loading model...')
        model = os.path.join('./Model', self.modelFile)
        self.saver.restore(sess,model)
        print ('Reading testing data...')
        test_sample = rd.read_data(self.testingDataFile)
        test_label  = rd.read_data(self.testingLabelFile)
        sampleNumber = len(test_label)
        print 'Sample Number is ', sampleNumber
        print ('Predicting data...')
        prediction = tf.argmax(self.y_conv, 1)
        begin = 0
        end = 0
        scale = len(test_sample)
        result = np.array([0 for row in range(sampleNumber)], dtype = 'int32')
        count = 0
        widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        timer = pb.ProgressBar(widgets=widgets, maxval= scale).start()
        times = 0
        while end < scale:
	        end = begin + self.batchNumberForTesting
	        batch_xs, batch_ys = rd.read_next_batch(test_sample, test_label,scale,begin, end, numFeatures)
	        if end < scale:
		        times = times + self.batchNumberForTesting
	        else:
		        times = times + (scale - begin)
                timer.update(times)
	        r_tmp = prediction.eval(feed_dict = {self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
	        for j in range(len(r_tmp)):
	 	        result[count] = r_tmp[j]
		        count = count + 1
	        begin = begin + self.batchNumberForTesting
        timer.finish() 
        print ('\nOutput the prediction results...')
        if not os.path.isdir("./Result"):
            	os.makedirs("./Result")
        target = open(os.path.join('./Result', self.outputFile), 'w')
        for i in range(len(result)):
	        if str(result[i]) == '0':
		        target.write('0')
	        else:
		        target.write('1')
	        target.write('\n')
        sess.close()
def main(argv=None):
    trainingDataFile = ""
    trainingLabelFile = ""
    testingDataFile = ""
    testingLabelFile = ""
    modelFile = ""
    outputFile = ""
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht:l:m:e:p:o:", ["help","input=","output="])
        except getopt.error, msg:
             raise Usage(msg)
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print __doc__
                sys.exit(0)
            elif opt in ("-t", "--training"):
                trainingDataFile = arg
            elif opt in ("-l", "--trainingLabel"):
                trainingLabelFile = arg
	    elif opt in ("-m", "--model"):
                modelFile = arg
            elif opt in ("-e", "--testing"):
                testingDataFile = arg
            elif opt in ("-p", "--testingLabel"):
                testingLabelFile = arg
            elif opt in ("-o", "--output"):
                outputFile = arg
        if(trainingDataFile == "" or trainingLabelFile == "" or modelFile == "" or testingDataFile == "" or testingLabelFile == "" or outputFile ==""):
            print __doc__
            sys.exit(0)
        d = DeepLearning(trainingDataFile,trainingLabelFile, modelFile,testingDataFile,testingLabelFile, outputFile)
        d.start_train()
        d.start_test()
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
if __name__ == "__main__":
    main()
