#! /usr/bin/Rscript
  
rm(list=ls(all=TRUE))

library(nnet);	

Args <- commandArgs();

trainPath <- Args[6];

testPath <- Args[7];

resultPath <- Args[8];
 
trainSample = read.csv(trainPath,header = TRUE);

rownumber <- nrow(trainSample);

if(rownumber > 0)
{
	nnmodel <- nnet(V1~., trainSample, size = 10, decay = 0.01, maxit = 1000, linout = T, trace = F);

	testSample = read.csv(testPath,header = TRUE);
	  
	testSampleNumber <- nrow(testSample);

	if(testSampleNumber > 0)
	{
		p = predict(nnmodel,testSample[2:10]);
		write(p,resultPath,ncolumns = 1,append = FALSE);
	}
}
