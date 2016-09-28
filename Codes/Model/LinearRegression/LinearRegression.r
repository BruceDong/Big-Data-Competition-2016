#! /usr/bin/Rscript
 
rm(list=ls(all=TRUE))	

Args <- commandArgs();
#cat("trainPath = ",Args[6],"\n");
trainPath <- Args[6];
#cat("testPath = ",Args[7],"\n");
testPath <- Args[7];
#cat("resultPath = ",Args[8],"\n");
resultPath <- Args[8];

#Import Training data 
trainSample = read.csv(trainPath,header = TRUE);
#Number of Training Sampls;
rownumber <- nrow(trainSample);
#cat(rownumber);
if(rownumber > 0)
{
	#cat(trainPath);
	#cat("\n");
	#cat("trainNumber:");
	#cat(rownumber);
	#cat("\n");
	#Training models, where columns from V2 to V10 contains features, and V1 is the labels 
	model <- lm(V1~V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10,data = trainSample);

	testSample = read.csv(testPath,header = TRUE);
	#Forecasting 
	testSampleNumber <- nrow(testSample);
	#cat(testSampleNumber);
	if(testSampleNumber > 0)
	{
		#cat(testPath);
		#cat("\n");
		#cat("testNumber:");
		#cat(testSampleNumber);
		#cat("\n");
		p = predict(model,testSample[2:10]);
		#cat(resultPath);

		write(p,resultPath,ncolumns = 1,append = FALSE);
	}
}
