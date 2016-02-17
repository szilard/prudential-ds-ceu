library(readr)
library(dplyr)
library(xgboost)
library(caret)

seed <- 2131

train = read_csv("prudential/data/prudential_train.csv")
test = read_csv("prudential/data/prudential_test.csv")

print(dim(train))
print(head(train, n=5))
print(dim(test))
print(head(test, n=5))

test$Response = 0

testId = test$Id
train$Id = test$Id = NULL

train[is.na(train)] <- -1
test[is.na(test)] <- -1

train$Product_Info_2_char = as.factor(substr(train$Product_Info_2, 1,1))
train$Product_Info_2_num = as.factor(substr(train$Product_Info_2, 2,2))
test$Product_Info_2_char = as.factor(substr(test$Product_Info_2, 1,1))
test$Product_Info_2_num = as.factor(substr(test$Product_Info_2, 2,2))

train$BMI_Age <- train$BMI * train$Ins_Age
test$BMI_Age <- test$BMI * test$Ins_Age


train$custom1 <- as.numeric(train$Medical_History_15 < 10.0)
train$custom1[is.na(train$custom1)] <- 0.0
test$custom1 <- as.numeric(test$Medical_History_15 < 10.0)
test$custom1[is.na(test$custom1)] <- 0.0


train <- train %>% mutate(mkcount = rowSums(.[grep("Medical_Keyword_1$", colnames(train)):grep("Medical_Keyword_48$", colnames(train))]))
test <- test %>% mutate(mkcount = rowSums(.[grep("Medical_Keyword_1$", colnames(test)):grep("Medical_Keyword_48$", colnames(test))]))


train$custom2 <- as.numeric(train$Product_Info_4 < 0.075)
test$custom2 <- as.numeric(test$Product_Info_4 < 0.075)

train$custom3 <- as.numeric(train$Product_Info_4 == 1)
test$custom3 <- as.numeric(test$Product_Info_4 == 1)

train$custom4 <- train$BMI * train$Product_Info_4
test$custom4 <- test$BMI * test$Product_Info_4

train$custom5 <- train$BMI * train$Medical_Keyword_3 + 0.5
test$custom5 <- test$BMI * test$Medical_Keyword_3 + 0.5




response <- train$Response
train$Response <- NULL

train$Medical_History_10 <- NULL
train$Medical_History_24 <- NULL

test$Medical_History_10 <- NULL
test$Medical_History_24 <- NULL

feature.names <- colnames(train)

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

dtrain<-xgb.DMatrix(data=data.matrix(train[,feature.names]),label=response, missing=NA)
watchlist<-list(val=dtrain,train=dtrain)

param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.05, # 0.06, #0.01,
                max_depth           =  6, #changed from default of 8
                subsample           = 0.8, # 0.7
                min_child_weight    = 25,
                colsample_bytree    = 0.7, # 0.7
                silent              = 0
)

set.seed(seed)
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, 
                    verbose             = 1,  
                    print.every.n       = 10,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

dtest<-xgb.DMatrix(data=data.matrix(test[,feature.names]), missing = NA)


submission <- data.frame(train)
pred <- predict(clf, dtrain)
submission$Response <- pred

optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun, preds = submission$Response, trues = response)

#optCuts$par <- c(1.8, 2.1, 3.3, 4.3, 5.3, 6.3, 6.75)

submission = read_csv("prudential/data/sample_submission.csv")
pred <- predict(clf, dtest)
submission$Response <- pred
submission$Response <- as.numeric(Hmisc::cut2(submission$Response, c(-Inf, optCuts$par, Inf)))
table(submission$Response)



head(submission)
write_csv(submission, "xg_final2.csv")





# submission = read_csv("prudential/data/sample_submission.csv")
# pred <- predict(clf, dtest)
# submission$Response <- pred
# 
# 
# 
# responseOrder <- c(rep.int(1, 1505), 
#                    rep.int(2, 992), 
#                    rep.int(3, 1541), 
#                    rep.int(4, 2117), 
#                    rep.int(5, 2244),
#                    rep.int(6, 2096),
#                    rep.int(7, 3150),
#                    rep.int(8, 6120))
# 
# 
# submission <- submission[order(submission$Response, decreasing = FALSE),]
# submission <- cbind(submission, responseOrder)
# 
# submission <- submission[order(submission$Id, decreasing = FALSE),]
# submission$Response <- NULL
# 
# colnames(submission) <- c("Id", "Response")
# table(submission$Response)
# write_csv(submission, "xg_final2.csv")
