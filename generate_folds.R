library(data.table)
library(ggplot2)
library(gridExtra)
source("R/plot_proportions.R")

train = fread("data/gender_age_train.csv")
test = fread("data/gender_age_test.csv")
split_rate = nrow(train) / (nrow(train) + nrow(test))
events = fread("data/events.csv")
events[, train := device_id %in% train$device_id]

to_plot = events[, list(count=.N, proportion=mean(train)), by=round(latitude, 1)]
plot_proportions(to_plot, true_prop=split_rate, log=TRUE)

to_plot = events[, list(count=.N, proportion=mean(train)), by=round(longitude, 1)]
plot_proportions(to_plot, true_prop=split_rate, log=TRUE)

event_cnts = events[, .N, by=c("device_id", "train")]
to_plot = event_cnts[, list(count=.N, proportion=mean(train)), by=N]
plot_proportions(to_plot, true_prop=split_rate, log=TRUE)

# Assign folds randomly by device_id
set.seed(123)
fold_cnt = 4
train[, fold := sample(rep(1:4, ceiling(nrow(train)/4)), size=nrow(train))]
train_events = merge(events, train[, list(device_id, fold)], by="device_id")

p = list()
for(i in 1:4){
    to_plot = train_events[, list(count=.N, proportion=mean(fold==i)), by=round(latitude, 1)]
    p[[i]] = plot_proportions(to_plot, true_prop=0.25, log=TRUE)
}
grid.arrange(p[[1]], p[[2]], p[[3]], p[[4]])

p = list()
for(i in 1:4){
    to_plot = train_events[, list(count=.N, proportion=mean(fold==i)), by=round(longitude, 1)]
    p[[i]] = plot_proportions(to_plot, true_prop=0.25, log=TRUE)
}
grid.arrange(p[[1]], p[[2]], p[[3]], p[[4]])

p = list()
event_cnts = train_events[, .N, by=c("device_id", "fold")]
for(i in 1:4){
    to_plot = event_cnts[, list(count=.N, proportion=mean(fold==i)), by=N]
    p[[i]] = plot_proportions(to_plot, true_prop=0.25, log=TRUE)
}
grid.arrange(p[[1]], p[[2]], p[[3]], p[[4]])

# Split of targets

train[, list(count=.N, proportion=mean(gender=="M")), by=fold]
train[, list(count=.N, proportion=mean(group=="M32-38")), by=fold]
