#Removing Existing objects and cleaning Environment
rm(list = ls())

#Setting Current working directory
setwd("D:/Amol_Data/Edwisor/Assignments/Project_1")

#Importing Required libraries
library(readr)
library(gridExtra)
library(corrgram)
library(caret)
library(tidyr)
library(rpart)
library(randomForest)
library(dplyr)
library(ggplot2)
library(data.table)
library(gbm)
library(usdm)
library(DMwR)

##Loading training dataset

df_train <- read_csv("train_cab.csv")

View(df_train)

#Structure of the dataset 
str(df_train)

#Dimensions of data
dim(df_train)

#Summaries of train data set

summary(df_train)



########################################################################  Missing Value Analysis ######################################################

missing_val = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] = "Missing_Percentage"
missing_val$Missing_Percentage = (missing_val$Missing_Percentage/nrow(df_train)) * 100

missing_val = missing_val[order(-missing_val$Missing_Percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

View(missing_val)


df_train$passenger_count[is.na(df_train$passenger_count)] = mean(df_train$passenger_count,na.rm = T)
df_train$fare_amount[is.na(df_train$fare_amount)] = mean(df_train$fare_amount,na.rm = T)

sum(is.na(df_train))

#split datetime_stamp value into year,month,day,weekday and hour for easy processing
date_time_split = function(df){
  
  df = separate(df,"pickup_datetime", c("Date","Time"), sep = " ")
  df = separate(df,"Date", c("Year","Month","Day"), sep = "-")
  df = separate(df,"Time", c("Hour"), sep = ":")
  print(sum(is.na(df)))
  df$Year = as.numeric(df$Year)
  df$Month = as.numeric(df$Month)
  df$Day = as.numeric(df$Day)
  df$Hour = as.numeric(df$Hour)
  df$Year[is.na(df$Year)] = mean(df$Year, na.rm = T)
  df$Month[is.na(df$Month)] = mean(df$Month, na.rm = T)
  df$Day[is.na(df$Day)] = mean(df$Day, na.rm = T)
  df$Hour[is.na(df$Hour)] = mean(df$Hour, na.rm = T)
  return(df)
}

df_train = date_time_split(df_train)

#############Setting limit to train data#################

df_train = df_train[((df_train['fare_amount'] >=0)&(df_train['fare_amount'] <=600))&
                    ((df_train['pickup_longitude'] > -79) &(df_train['pickup_longitude'] < -70))&
                    ((df_train['pickup_latitude'] >36) &(df_train['pickup_latitude']<45))&
                    ((df_train['dropoff_longitude']>-79) &(df_train['dropoff_longitude']))&
                    ((df_train['dropoff_latitude']>=36)&(df_train['dropoff_latitude']<45))&
                    ((df_train['passenger_count']>=1)&(df_train['passenger_count']<=7)),]

##########################################################################  Outlier Analysis #############################################################

##Boxplot for outlier visualization

num_index = sapply(df_train,is.numeric)
num_data = df_train[,num_index]
cnames = colnames(num_data)
for (i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(aes_string(y=(cnames[i]),group = 1), data = subset(df_train))+
    stat_boxplot(geom = "errorbar",width = 0.5) +
    geom_boxplot(outlier.color = "red", fill = "grey",outlier.shape = 18,
                 outlier.size = 1, notch=FALSE) +
    theme(legend.position = "bottom")+
    labs(y=cnames[i],X = "count")+
    ggtitle(paste("Box plot ",cnames[i])))
  
  
}

gridExtra::grid.arrange(gn2,gn3,ncol=2)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn4,gn5,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)

for(i in cnames){
  print(i)
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  print(length(val))
  df_train[,i][df_train[,i] %in% val] = NA
}

df_train <- data.frame(sapply(df_train, function(x) ifelse(is.na(x),mean(x,na.rm=TRUE),x)))
#num = sapply(df_train,is.numeric)


dim(df_train)
##################### Histogram #######################

hist(df_train$fare_amount)
hist(df_train$pickup_latitude)
hist(df_train$pickup_longitude)
hist(df_train$dropoff_latitude)
hist(df_train$dropoff_longitude)
hist(df_train$passenger_count)
hist(df_train$Year)
hist(df_train$Month)
hist(df_train$Day)
hist(df_train$Hour)


##################### Scatter plot #######################

scat1 = ggplot(df_train, aes(x=fare_amount,y=pickup_latitude,group=1)) +geom_point()
scat2 = ggplot(df_train, aes(x = fare_amount, y = Year, group = 1)) +  geom_point()
scat3 = ggplot(df_train, aes(x = fare_amount, y = pickup_longitude  , group = 1)) +  geom_point()
scat4 = ggplot(df_train, aes(x = fare_amount, y = dropoff_longitude  , group = 1)) +  geom_point()
scat5 = ggplot(df_train, aes(x = fare_amount, y = dropoff_latitude  , group = 1)) +  geom_point()
scat6 = ggplot(df_train, aes(x = fare_amount, y = passenger_count  , group = 1)) +  geom_point()
scat7 = ggplot(df_train, aes(x = fare_amount, y = Day  , group = 1)) +  geom_point()
scat8 = ggplot(df_train, aes(x = fare_amount, y = Hour  , group = 1)) +  geom_point()
scat9 = ggplot(df_train, aes(x = fare_amount, y = Month  , group = 1)) +  geom_point()


gridExtra::grid.arrange(scat1,scat2,scat3,scat4,scat5,scat6,scat7,scat8,scat9,ncol=3)



#####################################################################  Feature Selection #######################################################
##Correlation plot

corrgram(df_train[,num_index],order=F,
         upper.panel = panel.pie, text.panel = panel.txt, main = "correlation Plot")

vifcor(df_train)


######Splitting the train data into train and test set########

set.seed(12)
n = nrow(df_train)

train_index = sample(1:n, size = round(0.8*n),replace = FALSE)
train = df_train[train_index, ]
test = df_train[-train_index, ]
X_train = subset(train,select = -c(fare_amount))
y_train = subset(train,select = c(fare_amount))

X_test = subset(test,select = -c(fare_amount))
y_test = subset(test,select = c(fare_amount))

##################### Feature Scaling #######################

for(i in colnames(X_train))
{
  print(i)
  X_train[,i] = (X_train[,i] - min(X_train[,i]))/(max(X_train[,i])-min(X_train[,i]))
}

for(i in colnames(X_test))
{
  print(i)
  X_test[,i] = (X_test[,i] - min(X_test[,i]))/(max(X_train[,i]) - min(X_train[,i]))
  
}

###################################################################### Model Developement #########################################33333#####################


##################################  Linear Regression ###########################################

set.seed(100)
#Model
LR_model = lm(fare_amount~., data=train)
summary(LR_model)

##Prediction

LR_model_Prediction = predict(LR_model,test[,-1])

##Result

regr.eval(trues = y_test$fare_amount,preds=LR_model_Prediction,stats = c("mae","mse","rmse","mape"))

#mae        mse       rmse       mape 
#5.3697534 72.3076842  8.5033925  0.6104452 




#########################################  KNN ###################################################
#Model
KNN_model = knnreg(fare_amount~., data=c(X_train,y_train))

##Prediction

KNN_model_prediction = predict(KNN_model,X_test)
regr.eval(trues = y_test$fare_amount,preds = KNN_model_prediction,stats = c("mae","mse","rmse","mape"))

#mae        mse       rmse       mape 
#8.585506 131.221519  11.455196   1.079803 




#######################################  Decision Tree #############################################

#Model
DT_model = rpart(fare_amount~., data= train,method = 'anova')

#Prediction
DT_model_prediction = predict(DT_model,test[,-1])

##result

regr.eval(trues = y_test$fare_amount,preds = DT_model_prediction,stats = c("mae","mse","rmse","mape"))

#mae        mse       rmse       mape 
#4.0939139 36.2065533  6.0171882  0.4925124 



######################################  Random Forest ###############################################

#Model
RF_model = randomForest(fare_amount ~., data = train,ntree=500,nodesize = 8,importance=TRUE)

#Prediction
RF_model_prediction = predict(RF_model,test[,-1])

##Result
regr.eval(trues = y_test$fare_amount,preds = RF_model_prediction,stats = c("mae","mse","rmse","mape"))

#mae       mse      rmse      mape 
#2.400427 17.654299  4.201702  0.280484 


df_result = data.frame("rmse"=c("8.50","11.43","6.01","4.20"),"Model" = c("Linear Regression","KNN","Decision Tree","Random Forest"))
print(df_result)

#From Above table it is seen that Random forest has lowest RMSE Value.so, we will freeze Random forest for model deployment
###Testing of trained model on test data

##Loading Test data set
df_test <- read_csv("test.csv")

#Dimesions of test data
dim(df_test)

#Summary of test data
summary(df_test)


##Checking Missing values present in the test data
missing_val_test = data.frame(apply(df_test,2,function(x){sum(is.na(x))}))
missing_val_test$Columns = row.names(missing_val_test)
names(missing_val_test)[1] =  "Missing_percentage"
missing_val_test$Missing_percentage = (missing_val_test$Missing_percentage/nrow(df_test)) * 100
missing_val_test = missing_val_test[order(-missing_val_test$Missing_percentage),]
row.names(missing_val_test) = NULL
missing_val_test = missing_val_test[,c(2,1)]
View(missing_val_test)


#Divinding datetime stamp data into year,month,day hours
df_test = date_time_split(df_test)

##Running Trained random forest model on test data
RF_trained_model = predict(RF_model,df_test)

#saving the result into datframe
RF_prediction_result = data.frame("fare_amount" = RF_trained_model)


##Saving the predicted result to local machine
write.csv(RF_prediction_result,"Test_data_predicton_R.csv",row.names = FALSE)



