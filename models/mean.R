library(data.table)
library(bit64)

sapply(dir("R/", full.names=TRUE), source)

train = fread("data/gender_age_train_folds.csv")
test = fread("data/gender_age_test.csv")
phone_brand = unique(fread("data/phone_brand_device_model.csv"))
# Arbitrarily remove cases with two brands/models:
phone_brand[, row_id := .N, by=device_id]
phone_brand = phone_brand[row_id == 1, ]
phone_brand[, row_id := NULL]
# lose one row with a missing brand/model:
train = merge(train, phone_brand, by="device_id")
test = merge(test, phone_brand, by="device_id", all.x=TRUE)
test[is.na(phone_brand), ]

fit_model = function(X, y){
    train[, .N/nrow(train), group]
}
pred_model = function(model, X){
    pred_probs = matrix(model$V1, nrow=nrow(X), ncol=12, byrow = TRUE)
    colnames(pred_probs) = model$group
    return(pred_probs)
}
cv = crossValidation(model=list(predict=pred_model, fit=fit_model),
                    xTrain = train,
                    yTrain = factor(train$group), # not used!
                    xTest = test,
                    cvIndices = train[, fold])
summary(cv)
run(cv, metric=multi_class_log_loss, plotResults=TRUE, logged=FALSE)
