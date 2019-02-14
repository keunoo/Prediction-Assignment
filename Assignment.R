library(stringr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)

getwd()
setwd(dir="/Users/keunoo/R/Coursera")
rm(list=ls())

#read
data_full <- read.csv(file = "pml-training.csv")
data_test <- read.csv(file = "pml-testing.csv")
summary(data_test)

#preliminary adjusments
head(data_full)
dim(data_full)
summary(data_full)
colSums(is.na(data_full))
cc<- complete.cases(data_full)
sum(cc)
data_full$max_roll_belt
names(data_full)
sapply(data_full, class) 

numeric.vars <-names(data_full)[sapply(data_full, class) %in% c("integer", "numeric")] # get numeric var names
data_num <- data_full[, c(numeric.vars,"classe")] # get only numeric variables

valid.vars <-names(data_num)[colSums(is.na(data_num)) %in% c(0)] # get valid vars
data_valid <- data_num[, valid.vars] # get only valid variables

final.vars <- setdiff(colnames(data_valid),c("X","raw_timestamp_part_1","raw_timestamp_part_2"))
data_final <- data_valid[, final.vars] # get only final variables


summary(data_final)
dim(data_final)
head(data_final)

#correlation analysis
cor(data_final[, -c(54)])
cor.data <- data.frame(round(cor(data_final[, -c(54)]), 2)) 
cor.data
pairs(data_final[, c(1,2)])

## PLOT

ggplot(data=data_final, mapping = aes(x=classe)) + geom_bar()

dim(data_final)
for (i in c(1:20))
{
  plot <- ggplot(data=data_final, mapping = aes(x=classe, y = data_final[,i], fill = classe)) + geom_boxplot() + labs(x = colnames(data_final)[i])
  print(plot)
}

## Build models  

partition <- createDataPartition(data_final$classe, list = FALSE, p = .75)
data.train <- data_final[partition, ]
data.test <- data_final[-partition, ]

table(data.train$classe) / nrow(data.train)
table(data.test$classe) / nrow(data.test)

### Model 1 - Decision tree

#simple
dt <- rpart(classe ~ ., 
            data = data.train,
            control = rpart.control(minbucket = 5, cp = .02, maxdepth = 5),
            parms = list(split = "gini"))

rpart.plot(dt)

print("Training confusion matrix")
predicted <- predict(dt, type = "class")
confusionMatrix(predicted, factor(data.train$classe)) 

print("Test confusion matrix")
predicted <- predict(dt, newdata = data.test, type = "class")
confusionMatrix(predicted, factor(data.test$classe)) 

#complex
dt <- rpart(classe ~ ., 
            data = data.train,
            control = rpart.control(minbucket = 5, cp = .001, maxdepth = 10),
            parms = list(split = "gini"))

rpart.plot(dt)

print("Training confusion matrix")
predicted <- predict(dt, type = "class")
confusionMatrix(predicted, factor(data.train$classe)) 

print("Test confusion matrix")
predicted <- predict(dt, newdata = data.test, type = "class")
confusionMatrix(predicted, factor(data.test$classe)) 

### Model 2 - Random forest classification

control <- trainControl(method = "repeatedcv", 
                        number = 5, 
                        repeats = 2)

tune_grid <- expand.grid(mtry = c(20:40))

rf <- train(classe ~ ., 
            data = data.train,
            method = "rf",
            ntree = 50,
            importance = TRUE,
            trControl = control,
            tuneGrid = tune_grid)
plot(rf)
plot(varImp(rf), top = 15, main = "Variable Importance of Classification Random Forest")

print("Training confusion matrix")
predicted <- predict(rf, type = "raw")
predicted
confusionMatrix(predicted, factor(data.train$classe)) 

print("Test confusion matrix")
predicted <- predict(rf, newdata = data.test, type = "raw")
confusionMatrix(predicted, factor(data.test$classe)) 

#Test data
dim(data_test)

numeric.vars <-names(data_test)[sapply(data_test, class) %in% c("integer", "numeric")] # get numeric var names
numeric.vars
data.test.num <- data_test[, c(numeric.vars)] # get only numeric variables

valid.vars <-names(data.test.num)[colSums(is.na(data.test.num)) %in% c(0)] # get valid vars
data.test.valid <- data.test.num[, valid.vars] # get only valid variables

final.vars <- setdiff(colnames(data.test.valid),c("X","raw_timestamp_part_1","raw_timestamp_part_2"))
data.test.final <- data.test.valid[, final.vars] # get only final variables
dim(data.test.final)
dim(data_final)
colnames(data_final)
colnames(data.test.final)
data.test.final <- data.test.final[,-c(54)]

predicted <- predict(rf, newdata = data.test.final, type = "raw")
predicted
