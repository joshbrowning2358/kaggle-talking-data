library(data.table)
library(bit64)
library(xgboost)
library(Matrix)
library(ggplot2)

sapply(dir("R/", full.names=TRUE), source)

train = fread("data/gender_age_train_folds.csv")
phone_brand = unique(fread("data/phone_brand_device_model.csv"))
# Arbitrarily remove cases with two brands/models:
phone_brand[, row_id := .N, by=device_id]
phone_brand = phone_brand[row_id == 1, ]
phone_brand[, row_id := NULL]

train = merge(train, phone_brand, by="device_id", all.x=TRUE)
train[is.na(phone_brand), phone_brand := "NA"]
train[is.na(device_model), device_model := "NA"]

# Ensure factor levels are consistent
levels=c("M23-26", "M32-38", "M29-31", "F43+", "F27-28", "F29-32",
         "M22-", "M39+", "M27-28", "F33-42", "F23-", "F24-26")
y = as.numeric(factor(train$group, levels=levels)) - 1

X_sparse = sparse.model.matrix(fold ~ device_id + factor(phone_brand) +
                                   factor(device_model) + 0,
                               data=train)
dtrain = xgb.DMatrix(data=X_sparse, label=y)

params = merge(
    merge(data.frame(eta=c(0.1, 0.3, 1)),
          data.frame(max_depth=c(4, 6, 8))),
    data.frame(min_child_weight=c(1, 2, 4)))

results = NULL
for(eta in c(0.1, 0.3, 1)){
    for(max_depth in c(4, 6, 8)){
        for(min_child_weight in c(1, 2, 4)){
            model = xgb.cv(params=list(eta=eta, max_depth=max_depth,
                                       min_child_weight=min_child_weight, silent=1),
                           data=dtrain,
                           nrounds=1000,
                           watchlist=list(validation1=dval),
                           verbose=-1,
                           early.stop.round=3,
                           maximize=FALSE,
                           objective="multi:softprob",
                           num_class=12,
                           eval_metric="mlogloss", nfold=4)
            model[, iteration := 1:.N]
            model[, eta := eta]
            model[, max_depth := max_depth]
            model[, min_child_weight := min_child_weight]
            results = rbind(results, model)
        }
    }
}

ggplot(results[min_child_weight == 4, ], aes(x=iteration)) +
    geom_line(aes(y=train.mlogloss.mean, color="train")) +
    geom_line(aes(y=test.mlogloss.mean, color="test")) +
    facet_wrap(eta ~ max_depth, scale="free")
