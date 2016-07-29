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
installed_app_cnts = fread("features/installed_app_category_counts.csv")
active_app_cnts = fread("features/active_app_category_counts.csv")

merge_features = function(data){
    pb = merge(data, phone_brand[, list(device_id, phone_brand)], by="device_id", all.x=TRUE)
    setnames(pb, "phone_brand", "variable")
    pb[, value := 1]
    pb[is.na(variable), variable := "NA_phone_brand"]
    dm = merge(data, phone_brand[, list(device_id, device_model)], by="device_id", all.x=TRUE)
    setnames(dm, "device_model", "variable")
    dm[, value := 1]
    dm[is.na(variable), variable := "NA_device_model"]
    output = rbind(pb, dm)
    newdata = merge(data, installed_app_cnts, by="device_id", all.x=TRUE)
    setnames(newdata, c("category", "app_type_cnt"), c("variable", "value"))
    newdata[is.na(variable), value := 1]
    newdata[is.na(variable), variable := "no_app_event"]
    newdata[variable != "no_app_event", variable := paste0("inst_", variable)]
    output = rbind(output, newdata)
    newdata = merge(data, active_app_cnts, by="device_id")
    setnames(newdata, c("category", "app_type_cnt"), c("variable", "value"))
    newdata[, variable := paste0("inst_", variable)]
    output = rbind(output, newdata)
    return(output)
}

train = merge_features(train)
test = merge_features(test)

# Ensure factor levels are consistent
levels=c("M23-26", "M32-38", "M29-31", "F43+", "F27-28", "F29-32",
         "M22-", "M39+", "M27-28", "F33-42", "F23-", "F24-26")
variable_map = data.table(variable = unique(c(train$variable, test$variable)))
variable_map[, col := 1:.N - 1]

device_id_map = data.table(device_id = unique(train$device_id))
device_id_map[, row := 1:.N - 1]
fold = merge(train, device_id_map, by="device_id")
fold = unique(fold[, list(device_id, fold, row)])[order(row), ]
X = merge(train, device_id_map, by="device_id")
X = merge(X, variable_map, by="variable")
y = unique(X[, list(row, group)])[order(row), ]
y = as.numeric(factor(y$group, levels=levels)) - 1
X_sparse = sparseMatrix(i=X$row + 1, j=as.integer(X$col)+1, x=X$value,
                        dims=c(nrow(device_id_map), nrow(variable_map)))

device_id_map = data.table(device_id = unique(test$device_id))
device_id_map[, row := 1:.N - 1]
test_mat = merge(test, device_id_map, by="device_id")
test_mat = merge(test_mat, variable_map, by="variable")
test_mat = sparseMatrix(i=test_mat$row + 1, j=as.integer(test_mat$col)+1, x=test_mat$value,
                        dims=c(nrow(device_id_map), nrow(variable_map)))


fit_model = function(X, y){
    validation_index = sample(c(T, F), size=nrow(X), prob=c(0.3, 0.7), replace=TRUE)
    dtrain = xgb.DMatrix(data=X[!validation_index, ], label=y[!validation_index])
    dval = xgb.DMatrix(data=X[validation_index, ], label=y[validation_index])
    # dtrain = xgb.DMatrix(data=X_sparse, label=y)
    model = xgb.train(params = list(eta=0.03, max_depth=8,
                                    num_class=12, objective="multi:softprob",
                                    eval_metric="mlogloss"),
                      watchlist=list(validation=dval, train=dtrain),
                      data=dtrain,
                      early.stop.round = 10,
                      nrounds=5
                      )
    return(model)
}
pred_model = function(model, X){
    out = predict(model, newdata=X)
    out = matrix(out, ncol=12, byrow=TRUE)
    colnames(out) = levels
    return(out)
}
cv = crossValidation(model=list(predict=pred_model, fit=fit_model),
                    xTrain = X_sparse,
                    yTrain = y,
                    xTest = test_mat,
                    cvIndices = fold[, fold])
summary(cv)
run(cv, metric=multi_class_log_loss, plotResults=FALSE, logged=FALSE, idCol="device_id",
    filename="data/models/xgboost_brand_time_app")
run(cv, metric=multi_class_log_loss, plotResults=FALSE, logged=FALSE,
    filename="data/models/xgboost_brand_time_app")
