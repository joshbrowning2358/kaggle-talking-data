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

fitModel = function(X, y){
    gender_model = glm(factor(gender) ~ phone_brand, data=X, family="binomial")
    age_model = glm(age ~ phone_brand, data=X)
    gender_missing_model = X[, mean(gender=="F")]
    age_missing_model = X[, list(mu=mean(age), sd=sd(age))]
    return(list(gender_model, age_model, gender_missing_model, age_missing_model,
                X[, unique(phone_brand)]))
}
predModel = function(model, X){
    missing = X[, !phone_brand %in% model[[5]]]
    pred = data.table(gender_male=rep(NA_real_, nrow(X)), gender_female=rep(NA_real_, nrow(X)),
                      age_fit=rep(NA_real_, nrow(X)), age_sd=rep(NA_real_, nrow(X)))
    
    gender_pred_avail = predict(gender_model, newdata=X[!missing, ], type="response")
    pred[!missing, gender_male := 1 - gender_pred_avail]
    pred[!missing, gender_female := gender_pred_avail]
    pred[missing, gender_male := 1 - model[[3]]]
    pred[missing, gender_female := model[[3]]]
    
    age_pred_avail = predict(age_model, newdata=X[!missing, ], se.fit=TRUE)
    pred[!missing, age_fit:= age_pred_avail$fit]
    pred[!missing, age_sd := age_pred_avail$se.fit]
    pred[missing, age_fit := model[[4]]$mu]
    pred[missing, age_sd := model[[4]]$sd]
}
cv = crossValidation(model=list(predict=predModel, fit=fitModel),
                    xTrain = train,
                    yTrain = train[, Demanda_uni_equil],
                    xTest = test,
                    cvIndices = train[, Semana])
summary(cv)
run(cv, metric=rmsle, plotResults=FALSE, logged=FALSE)
