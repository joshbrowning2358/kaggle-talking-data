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
apps_per_event = fread("features/apps_per_event.csv")
avg_position = fread("features/avg_position.csv")
avg_position[avg_latitude == 0 | avg_latitude == 1, avg_latitude := NA]
avg_position[avg_longitude == 0 | avg_longitude == 1, avg_longitude := NA]
sd_position = fread("features/sd_position.csv")
count_by_hour = fread("features/count_by_hour.csv")
count_by_period = fread("features/count_by_period.csv")
event_counts = fread("features/event_counts.csv")

merge_features = function(data){
#     data = merge(data, phone_brand, by="device_id", all.x=TRUE)
#     data[is.na(phone_brand), phone_brand := "NA"]
#     data[is.na(device_model), device_model := "NA"]
    data = merge(data, apps_per_event, by="device_id")
    data = merge(data, avg_position, by="device_id")
    data = merge(data, sd_position, by="device_id")
    data = merge(data, count_by_hour, by="device_id")
    data = merge(data, count_by_period, by="device_id")
    data = merge(data, event_counts, by="device_id")
    return(data)
}

train = merge_features(train)
test = merge_features(test)

# Ensure factor levels are consistent
levels=c("M23-26", "M32-38", "M29-31", "F43+", "F27-28", "F29-32",
         "M22-", "M39+", "M27-28", "F33-42", "F23-", "F24-26")
brands = phone_brand[, c(unique(phone_brand), "NA")]
models = phone_brand[, c(unique(device_model), "NA")]

var_cols = colnames(train)[6:ncol(train)]

fit_model = function(X, y){
    y = as.numeric(factor(y, levels=levels)) - 1
    validation_index = sample(c(T, F), size=nrow(X), prob=c(0.3, 0.7), replace=TRUE)
    dtrain = xgb.DMatrix(data=as.matrix(X[!validation_index, var_cols, with=FALSE]),
                         label=y[!validation_index], missing=NA)
    dval = xgb.DMatrix(data=as.matrix(X[validation_index, var_cols, with=FALSE]),
                       label=y[validation_index], missing=NA)
    # dtrain = xgb.DMatrix(data=X_sparse, label=y)
    model = xgb.train(params = list(eta=0.02, max_depth=8,
                                    num_class=12, objective="multi:softprob",
                                    eval_metric="mlogloss"),
                      watchlist=list(validation=dval, train=dtrain),
                      data=dtrain,
                      early.stop.round = 10,
                      nrounds=500
                      )
    return(model)
}
pred_model = function(model, X){
    out = predict(model, newdata=X[, var_cols, with=FALSE])
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
run(cv, metric=multi_class_log_loss, plotResults=FALSE, logged=FALSE, idCol="device_id",
    filename="data/models/xgboost_brand_time_location")