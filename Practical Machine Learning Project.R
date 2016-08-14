#Six young health participants were asked to perform one set of 10 repetitions of the
#Unilateral Dumbbell Biceps Curl in five different fashions: 
#1. exactly according to the specification (Class A)
#2. throwing the elbows to the front (Class B)
#3. lifting the dumbbell only halfway (Class C)
#4. lowering the dumbbell only halfway (Class D)
#5. throwing the hips to the front (Class E).
#Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4Fu8cOtIO

#The goal of this project is to predict the manner in which they did the exercise

library(caret)
training <- read.csv("~/Practical Machine Learning/Course_project/pml-training.csv")
testing <- read.csv("~/Practical Machine Learning/Course_project/pml-testing.csv")

#a) EXPLORATORY DATA ANALYSIS
length(names(training)) 
# the dataset contains 160 variables; variable Classe describes the manner they did the exercise
# the purpose of this project is to predict this variable Classe (i.e., A, B, C, D or E).
summary(training)
#this shows that there are a number of NAs in the data set 
sum(is.na(training)) # 1287472 NAs
sum(is.na(testing))  #2000 NAs
#we may imput the missing values using nearest neighbors with impute.knn(datamatrix)
#in caret, preProcess(method="knnImpute") can be used

#The last column in training is 'Classe' and in testing it is 'problem_id'
#Excluding the last columns, verify that all column names in the training and testing datasets are identical
colnames_train <- colnames(training)
colnames_test <- colnames(testing)
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_test)-1])

#b) PREPREOCESSING
#the testing set contains only 20 observations while the training set contains 19622 observations
#partition the training set into training and testing sets for cross-validation
#The final selected model will be applied on the testing set of 20 observations
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
raw_training_cv <- training[inTrain, ]
raw_testing_cv <- training[-inTrain,]

#remove covariates with near-zero-variablity 
nzv <- nearZeroVar(raw_training_cv, saveMetrics=TRUE)
raw_training_cv <- raw_training_cv[ , nzv$nzv==FALSE]
sum(nzv$nzv==TRUE) #variables are removed because they have near-zero-variablity and do not affect the prediction much
raw_testing_cv <- raw_testing_cv[ ,nzv$nzv==FALSE]

#handling NAs:keep only variables without NAs
training_cv <- raw_training_cv[ , colSums(is.na(raw_training_cv))==0]
#remove the first six columns of nonNA_training_cv which are row numbers and time series info
#the data analysis has no time-dependence
training_cv <- training_cv[c(-1,-2,-3,-4,-5,-6)]
#transform testing_cv and testing datasets so that they contain only variables in training_cv
testing_cv <- raw_testing_cv[ , colnames(training_cv)] #same colnames as training_cv
testing <- testing[ , colnames(training_cv[ ,-53])] #same as training_cv with last column removed

#which features have high correlation (>0.9)
M <- cor(training_cv[ , c(-58)]) #excluding the Classe variable
which(M>0.9, arr.ind=TRUE)

#TRAINing and PREDICTION WITH DECISION TREES
library(randomForest)
library(rattle)
library(rpart.plot)
set.seed(12345)
Fit_trees <- train(classe~., data = training_cv, method="rpart")
fancyRpartPlot(Fit_trees$finalModel)
Prediction_trees <- predict(Fit_trees, testing_cv)
conf_mat <- confusionMatrix(Prediction_trees, testing_cv$classe )
conf_mat
#TRAINING and PREDICTION with RANDOMFORESTS
set.seed(12345)
Fit_rf <- train(classe~., data = training_cv, method = "rf")
Prediction_rf <- predict(Fit_rf, testing_cv)
confusionMatrix(Prediction_rf, testing_cv$classe)
#took about 3 hours. Save the result Fit_rf object for later use, say in reporting
save(Fit_rf, file="Fit_rf.RData")

#Varible importance in the model: the firts five variables of high relative importance
Var_imp <- varImp(Fit_rf)$importance
Var_imp[head(order(Var_imp, decreasing=TRUE), 5L), , drop=FALSE] 
#With out drop=False, only the values are shown with out their corresponding variable names

#PREDICTION on the 20-sample TEST SET
#load the saved training model result object
load(file = "Fit_rf.RData", verbose = TRUE)
prediction_onTEST <- predict(Fit_rf, testing)
prediction_onTEST
