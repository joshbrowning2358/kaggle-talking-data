library(data.table)
library(bit64)
library(lme4)

sapply(dir("R/", full.names=TRUE), source)

train = fread("data/gender_age_train_folds.csv")
test = fread("data/gender_age_test.csv")
phone_brand = unique(fread("data/phone_brand_device_model.csv"))
# Arbitrarily remove cases with two brands/models:
phone_brand[, row_id := .N, by=device_id]
phone_brand = phone_brand[row_id == 1, ]
phone_brand[, row_id := NULL]

train = merge(train, phone_brand, by="device_id", all.x=TRUE)
train[is.na(phone_brand), phone_brand := "NA"]
train[is.na(device_model), device_model := "NA"]
test = merge(test, phone_brand, by="device_id", all.x=TRUE)
test[is.na(phone_brand), phone_brand := "NA"]
test[is.na(device_model), device_model := "NA"]

# Ensure factor levels are consistent
levels=c("M23-26", "M32-38", "M29-31", "F43+", "F27-28", "F29-32",
         "M22-", "M39+", "M27-28", "F33-42", "F23-", "F24-26")

fit_model = function(X, y){
    model_gender = glmer(gender=="M" ~ (1|phone_brand) + (1|device_model), data=X,
                         family="binomial")
    model_age = lmer(age ~ (1|phone_brand) + (1|device_model), data=X)
    return(list(model_gender, model_age))
}
pred_model = function(model, X){
    gender_est = predict(model[[1]], X)
    age_est = predict(model[[2]], X)
    
}
cv = crossValidation(model=list(predict=pred_model, fit=fit_model),
                    xTrain = train,
                    yTrain = factor(train$group, levels=levels),
                    xTest = test,
                    cvIndices = train[, fold])
summary(cv)
run(cv, metric=multi_class_log_loss, plotResults=FALSE, logged=TRUE, filename="xgboost")
