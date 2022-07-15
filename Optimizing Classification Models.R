# Digital Analytics Project - Otto Group Classification Challenge (30.11.2020)

# Setup
# Installing the necessary libraries #install.packages("DMwR") #install.packages("Caret") #install.packages("doParallel") #install.packages("ggplot")

# Setting working directory and reading in data setwd("C:/Users/David/Desktop/Class Project") data = read.csv("ClassifyProducts.csv")

# Exploratory data analysis str(data) table(data$target)

ggplot(data = data, aes(x = target)) + geom_bar(fill="steelblue")+
  ggtitle("Distribution of Target Variable")+ theme_minimal()

summary(data)

# Data Preparation

# Check for missing data
colSums(is.na(data))

# Removing first row (product ID is of no value to us)
data <- data[,-1]

# Setting target variable as factor
data$target <- as.factor(data$target)

# Checking for high correlation, recommended threshold is 0.9
library(caret)
correlation.mat<-cor(data[,-94])
highcorrelation <- findCorrelation(correlation.mat, 0.85) highcorrelation

# Normalization of variables/features
preProcValues <- preProcess(data, method = "range") data <- predict(preProcValues, data)
summary(data)

# Creating sample sample <- data dim(sample)

# Creating test and training data sets
set.seed(54321)
index <- createDataPartition(sample$target, p=0.8, list=FALSE) training <- sample[index,]
test <- sample[-index,] nrow(training) nrow(test)

# Applying downsampling to training set due to data set imbalance
library(caret) set.seed(54321)
downsample <- downSample(training[, -ncol(training)], training$target, yname
                         = "target") summary(downsample$target) training<-downsample

# Summary of finished test and training sets summary(training$target) summary(test$target)

# Classification

# Setup of report and cross validation
TControl <- trainControl(method="cv", number=5)
report <- data.frame(Model=character(), Acc.Train=numeric(), Acc.Test=numeric())

# K Nearest neighbors
set.seed(54321)
knnmodel <- train(target~., data=training, method="knn", trControl=TControl) knnmodel
prediction.test <- predict(knnmodel, test[,-94],type="raw") accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="k-NN", Acc.Train=0.00, Acc.Test=accte$overall['Accuracy']))
report


# Naive Bayes Model
set.seed(54321)
nbmodel <- train(target~., data=training, method="nb", trControl=TControl) nbmodel
prediction.train <- predict(nbmodel, training[,-94],type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="Naive Bayes", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


# C5.0 Model
set.seed(54321)
c5model <- train(target ~., data=training, method="C5.0", trControl=TControl) c5model
prediction.train <- predict(c5model, training[,-94], type="raw") prediction.test <- predict(c5model, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="C5.0", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy'])) report


# Cart Model
set.seed(54321)
cartmodel <- train(target ~., data=training, method="rpart", trControl=TControl)
cartmodel
prediction.train <- predict(cartmodel,training[,-94], type="raw") prediction.test <- predict(cartmodel,test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="CART", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


# Random Forest
set.seed(54321)
rformodel <- train(target ~., data=training, method="rf", trControl=TControl) rformodel
prediction.train <- predict(rformodel, training[,-94], type="raw") prediction.test <- predict(rformodel, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="Random Forest", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy'])) report


# Neural Network Model
set.seed(54321)
nnmodel <- train(target ~., data=training, method="nnet", trControl=TControl) nnmodel
prediction.train <- predict(nnmodel, training[,-94], type="raw") prediction.test <- predict(nnmodel, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="Neural Network", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


# Support Vector Machine model
set.seed(54321)
svmmodel.l <- train(target ~., data=training, method="svmLinear", trControl=TControl)
svmmodel.l
prediction.train <- predict(svmmodel.l, training[,-94], type="raw") prediction.test <- predict(svmmodel.l, test[,-94], type="raw")
acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="SVM (Linear)", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


# Radial Support Vector Machine Model
set.seed(54321)
svmmodel.r <- train(target ~., data=training, method="svmRadial", trControl=TControl)
svmmodel.r
prediction.train <- predict(svmmodel.r, training[,-94], type="raw") prediction.test <- predict(svmmodel.r, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="SVM (Radial)", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


# Comparing Models
results <- resamples(list(KNN=knnmodel, NBayes=nbmodel, C5.0=c5model,CART=cartmodel, RFor=rformodel,
                          NeuNet=nnmodel, SVM.L=svmmodel.l,
                          SVM.R=svmmodel.r)) summary(results) dotplot(results) report


# Tuning Radial Support Vector Machine
library(doParallel) library(caret)
cores <- makeCluster(detectCores()-1) registerDoParallel(cores = cores)
svmgrid <- expand.grid(sigma = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
                       C = c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2))
set.seed(54321)
tunedsvm.r <- train(target ~., data=training, method="svmRadial",
                    trControl=TControl, tuneGrid = svmgrid) tuningresultssvm.r <- data.frame(tunedsvm.r[["results"]])

# Plotting Accuracies across multiple parameter combinations
ggplot(tuningresultssvm.r, aes(x = C, y = Accuracy, color= as.factor(sigma)))
+
  geom_point()+
  scale_x_continuous(name="C parameter",) + scale_y_continuous(name="Accuracy",) +
  ggtitle("Accuracy Across Multiple Values for C and Sigma")+ theme_bw()

# Print the best tuning parameter sigma and C that maximizes model accuracy
tunedsvm.r$bestTune
stopCluster(cores)


# Tuning of Random Forrest

# Tuning across multiple parameters requires allot of computation. The "doparallel" libary allows for better multicore processing, speeding up computing times
library(doParallel) library(caret)
cores <- makeCluster(detectCores()-1) registerDoParallel(cores = cores)


# Test for optimal mtry and Ntree parameter
set.seed(54321)
mtrygrid <- expand.grid(mtry = c(3:20))
rformodelmtryn25 <- train(target ~., data=training, method="rf",metric = "Accuracy", tuneGrid=mtrygrid, trControl = TControl, importance = TRUE,ntree=25)
rformodelmtryn50 <- train(target ~., data=training, method="rf",metric = "Accuracy", tuneGrid=mtrygrid, trControl = TControl, importance = TRUE,ntree=50)
rformodelmtryn75 <- train(target ~., data=training, method="rf",metric = "Accuracy", tuneGrid=mtrygrid, trControl = TControl, importance = TRUE,ntree=75)
rformodelmtryn100 <- train(target ~., data=training, method="rf",metric = "Accuracy", tuneGrid=mtrygrid, trControl = TControl, importance = TRUE,ntree=100)
rformodelmtryn125 <- train(target ~., data=training, method="rf",metric = "Accuracy", tuneGrid=mtrygrid, trControl = TControl, importance = TRUE,ntree=125)
rformodelmtryn150 <- train(target ~., data=training, method="rf",metric = "Accuracy", tuneGrid=mtrygrid, trControl = TControl, importance = TRUE,ntree=150)


# Storing the results in dataframes for plotting and adding a Variable to indicate source model
tuningresultsmtryn25 <- data.frame(rformodelmtryn25[["results"]]) tuningresultsmtryn25$ntree <- 'Ntree 25'
tuningresultsmtryn50 <- data.frame(rformodelmtryn50[["results"]]) tuningresultsmtryn50$ntree <- 'Ntree 50'
tuningresultsmtryn75 <- data.frame(rformodelmtryn75[["results"]]) tuningresultsmtryn75$ntree <- 'Ntree 75'
tuningresultsmtryn100 <- data.frame(rformodelmtryn100[["results"]]) tuningresultsmtryn100$ntree <- 'Ntree 100'
tuningresultsmtryn125 <- data.frame(rformodelmtryn125[["results"]]) tuningresultsmtryn125$ntree <- 'Ntree 125'
tuningresultsmtryn150 <- data.frame(rformodelmtryn150[["results"]]) tuningresultsmtryn150$ntree <- 'Ntree 150'
plotdata <- rbind.data.frame(tuningresultsmtryn25, tuningresultsmtryn50, tuningresultsmtryn75, tuningresultsmtryn100, tuningresultsmtryn125, tuningresultsmtryn150)
stopCluster(cores)


# Plotting model accuracies across different Values for .mtry and ntree
library(caret)
ggplot(plotdata, aes(x = mtry, y = Accuracy))+ geom_line(aes(colour = ntree), size = 1.5)+ scale_x_continuous(name="Mtry Parameter",) + scale_y_continuous(name="Accuracy",) +
  ggtitle("Accuracy Across Multiple Values for Mtry and Ntree")+ theme_bw()

# best .mtry = 7 .ntree = 150

# Maxnode Tuning

#Different Values for maxnodes parameter to reduce overfitting with mtry and ntree held constant

library(caret)
tunedrfmn500 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=500)
prediction.train <- predict(tunedrfmn500, training[,-94], type="raw")
prediction.test <- predict(tunedrfmn500, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
Maxnodereport <- rbind(Maxnodereport, data.frame(Model="tunedrfmn500", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


Maxnodereport <- data.frame(Model=character(), Acc.Train=numeric(), Acc.Test=numeric())
tunedrfmn750 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=750)
prediction.train <- predict(tunedrfmn750, training[,-94], type="raw") prediction.test <- predict(tunedrfmn750, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
Maxnodereport <- rbind(Maxnodereport, data.frame(Model="tunedrfmn750", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


tunedrfmn1000 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=1000)
prediction.train <- predict(tunedrfmn1000, training[,-94], type="raw") prediction.test <- predict(tunedrfmn1000, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
Maxnodereport <- rbind(Maxnodereport, data.frame(Model="tunedrfmn1000", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))
tunedrfmn1250 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=1250)
prediction.train <- predict(tunedrfmn1250, training[,-94], type="raw") prediction.test <- predict(tunedrfmn1250, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
Maxnodereport <- rbind(Maxnodereport, data.frame(Model="tunedrfmn1250", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

tunedrfmn1500 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=1500)
prediction.train <- predict(tunedrfmn1500, training[,-94], type="raw") prediction.test <- predict(tunedrfmn1500, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
Maxnodereport <- rbind(Maxnodereport, data.frame(Model="tunedrfmn1500", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))


tunedrfmn2000 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=2000)
prediction.train <- predict(tunedrfmn2000, training[,-94], type="raw") prediction.test <- predict(tunedrfmn2000, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
Maxnodereport <- rbind(Maxnodereport, data.frame(Model="tunedrfmn2000", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

Maxnodereport

#plotting results
ggplot(Maxnodereport, aes(maxnodes))+ geom_line(aes(y=Acc.Train, color = "Acc.Train"), size = 1.5)+ geom_line(aes(y= Acc.Test, color = "Acc.Test"), size = 1.5)+ scale_x_continuous(name="maxnodes",) + scale_y_continuous(name="Accuracy",) +
  ggtitle("Test and Train Accuracy Across Differnt Values For Maxnodes")+ theme_bw()

# rerunning optimal model to attain confusion matrix and adding accuracy to inital report
tunedrfmn500 <- train(target ~., data=training, method="rf", trControl=TControl, .mtry=7, ntree=150,maxnodes=500)
prediction.train <- predict(tunedrfmn500, training[,-94], type="raw") prediction.test <- predict(tunedrfmn500, test[,-94], type="raw") acctr <- confusionMatrix(prediction.train, training[,94]) acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="Tuned Random Forrest", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

# plotting confusion matrix
library(ggplot2)
cm_d <- as.data.frame(accte$table)
ggplot(data = cm_d, aes(x = Prediction , y = Reference, fill = Freq))+ geom_tile() +
  geom_text(aes(label = paste(Freq)), color = 'white', size = 3) + theme_light() +
  guides(fill=FALSE)


# Applying other resampling Methods

#applying Upsampling
trainingUP <- sample[index,]
trainingUP <- upSample(trainingUP[, -ncol(trainingUP)], trainingUP$target, yname = "target")
nrow(trainingUP)

#applying Synthetic Minority Over-sampling Technique
library(DMwR)
trainingSMOTEs <- sample[index,]
trainingSMOTE <- SMOTE(target ~ ., data = trainingSMOTEs, perc.over=200, perc.under=400)

# check for missing values potentially created through SMOTE
colSums(is.na(trainingSMOTE))

#compare different training sets
nrow(training) nrow(trainingUP) nrow(trainingSMOTE)

# Random forest trained on SMOTE data, trying 15 random values for .mtry
set.seed(54321)
rformodelSMOTE <- train(target ~., data=trainingSMOTE, method="rf", trControl=TControl,importance = TRUE, ntree=150, tuneLength = 15)

#plotting results
library(caret)
ggplot(rformodelSMOTE, aes(x = mtry, y = Accuracy))+ geom_line()+
  scale_x_continuous(name="Mtry Parameter",) + scale_y_continuous(name="Accuracy",) +
  ggtitle("Rfor Model accuracy across .mtry values and Trained on SMOTE Data")+
  theme_bw()

#applying tuned SMOTE - rformodel to test data
set.seed(54321)
rformodelSMOTE <- train(target ~., data=trainingSMOTE, method="rf", trControl=TControl,importance = TRUE, ntree=150, .mtry=21) prediction.train <- predict(rformodel, trainingSMOTE[,-94], type="raw") prediction.test <- predict(rformodel, test[,-94], type="raw")
acctr <- confusionMatrix(prediction.train, trainingSMOTE[,94])
acctr$overall['Accuracy']
accte <- confusionMatrix(prediction.test, test[,94]) accte$overall['Accuracy']
accte$table
report <- rbind(report, data.frame(Model="Tuned Random Forest SMOTE", Acc.Train=acctr$overall['Accuracy'], Acc.Test=accte$overall['Accuracy']))

# plotting confusion matrix
library(ggplot2)
cm_s <- as.data.frame(accte$table)
ggplot(data = cm_s, aes(x = Prediction , y = Reference, fill = Freq))+ geom_tile() +
  geom_text(aes(label = paste(Freq)), color = 'white', size = 3) + theme_light() +
  guides(fill=FALSE)

#Taking a final look at the report
report

#end
