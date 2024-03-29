library(data.table)
library(bit64)
library(xgboost)
library(Matrix)

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
train_sparse = sparse.model.matrix(fold ~ device_id + factor(phone_brand) +
                                       factor(device_model) + 0,
                                   data=train)
test[, dummy := 1]
test_sparse = sparse.model.matrix(dummy ~ device_id + factor(phone_brand) +
                                       factor(device_model) + 0,
                                   data=test)

# Ensure factor levels are consistent
levels=c("M23-26", "M32-38", "M29-31", "F43+", "F27-28", "F29-32",
         "M22-", "M39+", "M27-28", "F33-42", "F23-", "F24-26")
brands = phone_brand[, c(unique(phone_brand), "NA")]
models = phone_brand[, c(unique(device_model), "NA")]

fit_model = function(X, y){
    y = as.numeric(factor(y, levels=levels)) - 1
    X_sparse = sparse.model.matrix(fold ~ device_id + factor(phone_brand, levels=brands) +
                                       factor(device_model, levels=models) + 0,
                                   data=X)
    # validation_index = sample(c(T, F), size=nrow(X), prob=c(0.3, 0.7), replace=TRUE)
    # dtrain = xgb.DMatrix(data=X_sparse[!validation_index, ], label=y[!validation_index])
    # dval = xgb.DMatrix(data=X_sparse[validation_index, ], label=y[validation_index])
    dtrain = xgb.DMatrix(data=X_sparse, label=y)
    model = xgb.train(params = list(eta=0.01, max_depth=8,
                                    num_class=12, objective="multi:softprob",
                                    eval_metric="mlogloss"),
                      #watchlist=list(validation=dval, train=dtrain),
                      data=dtrain,
                      #early.stop.round = 10
                      nrounds=500
                      )
    return(model)
}
pred_model = function(model, X){
    X[, dummy := 1]
    X_sparse = sparse.model.matrix(dummy ~ device_id + factor(phone_brand, levels=brands) +
                                       factor(device_model, levels=models) + 0,
                                   data=X)
    out = predict(model, newdata=X_sparse)
    out = matrix(out, ncol=12, byrow=TRUE)
    colnames(out) = levels
    return(out)
}
cv = crossValidation(model=list(predict=pred_model, fit=fit_model),
                    xTrain = train,
                    yTrain = factor(train$group, levels=levels),
                    xTest = test,
                    cvIndices = train[, fold])
summary(cv)
run(cv, metric=multi_class_log_loss, plotResults=FALSE, logged=TRUE, idCol="device_id",
    filename="data/models/xgboost_no_validation")