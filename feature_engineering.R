###################################################################################################
## Load the libraries and data
###################################################################################################

library(data.table)
library(Matrix)
library(bit64)
library(ggplot2)

train = fread("data/gender_age_train.csv")
test = fread("data/gender_age_test.csv")
phone_brand = fread("data/phone_brand_device_model.csv")
events = fread("data/events.csv")
app_events = fread("data/app_events.csv")
app_labels = fread("data/app_labels.csv")
label_categories = fread("data/label_categories.csv")

## Brand and model

brand = sparse.model.matrix(device_model ~ device_id + factor(phone_brand) + 0, data=phone_brand)
save(brand, file="features/brand.RData")

model = sparse.model.matrix(device_id ~ device_model + 0, data=phone_brand)
save(model, file="features/model.RData")

## Events

# counts
train[, has_event := device_id %in% events$device_id]
test[, has_event := device_id %in% events$device_id]
write.csv(rbind(train[, c("device_id", "has_event"), with=FALSE],
                test[, c("device_id", "has_event"), with=FALSE]), file="features/has_event.csv", row.names=FALSE)
event_counts = events[, list(event_counts=.N), by=device_id]
event_counts = rbind(data.table(device_id=train[!(has_event), device_id], event_counts=0),
                     event_counts)
event_counts = rbind(data.table(device_id=test[!(has_event), device_id], event_counts=0),
                     event_counts)
event_counts = event_counts[device_id %in% c(test$device_id, train$device_id), ]
write.csv(event_counts, file="features/event_counts.csv", row.names=FALSE)

# geo location
avg_position = events[, list(avg_latitude=mean(latitude), avg_longitude=mean(longitude)),
                      by = device_id]
avg_position = rbind(data.table(device_id=train[!(has_event), device_id], avg_latitude=NA, avg_longitude=NA),
                     avg_position)
avg_position = rbind(data.table(device_id=test[!(has_event), device_id], avg_latitude=NA, avg_longitude=NA),
                     avg_position)
avg_position = avg_position[device_id %in% c(test$device_id, train$device_id), ]
write.csv(avg_position, file="features/avg_position.csv", row.names=FALSE)

sd_position = events[, list(sd_latitude=sd(latitude), sd_longitude=sd(longitude)),
                      by = device_id]
sd_position = rbind(data.table(device_id=train[!(has_event), device_id], sd_latitude=NA, sd_longitude=NA),
                     sd_position)
sd_position = rbind(data.table(device_id=test[!(has_event), device_id], sd_latitude=NA, sd_longitude=NA),
                     sd_position)
sd_position = sd_position[device_id %in% c(test$device_id, train$device_id), ]
write.csv(sd_position, file="features/sd_position.csv", row.names=FALSE)

# time
events[, timestamp := as.POSIXct(timestamp)]
ggsave("event_counts_over_time.png",
    qplot(events[, as.POSIXct(as.character(timestamp, "%H:%M:%S"), format="%H:%M:%S")]) +
        labs(x= "Event hour (date is meaningless)"))
events[, hour_of_day := as.numeric(as.character(timestamp, "%H"))]
count_by_hour = events[, .N, by=c("device_id", "hour_of_day")]
count_by_hour = dcast(count_by_hour, device_id ~ hour_of_day, value.var="N", fill=0)
setnames(count_by_hour, as.character(0:23), paste0("hour_", 0:23, "_cnt"))
count_by_hour = rbindlist(list(train[!(has_event), list(device_id)], count_by_hour), fill=TRUE)
count_by_hour = rbindlist(list(test[!(has_event), list(device_id)], count_by_hour), fill=TRUE)
sapply(colnames(count_by_hour), function(col){
    count_by_hour[is.na(get(col)), c(col) := 0]
})
count_by_hour = count_by_hour[device_id %in% c(test$device_id, train$device_id), ]
write.csv(count_by_hour, "features/count_by_hour.csv", row.names=FALSE)

count_by_period = events[, list(early_morning_cnt = sum(hour_of_day <= 5),
                                 morning_cnt = sum(hour_of_day > 5 & hour_of_day <= 12 ),
                                 afternoon_cnt = sum(hour_of_day > 12 & hour_of_day <= 18),
                                 evening_cnt = sum(hour_of_day > 18)), by=device_id]
count_by_period = rbindlist(list(train[!(has_event), list(device_id)], count_by_period), fill=TRUE)
count_by_period = rbindlist(list(test[!(has_event), list(device_id)], count_by_period), fill=TRUE)
sapply(colnames(count_by_period), function(col){
    count_by_period[is.na(get(col)), c(col) := 0]
})
count_by_period = count_by_period[device_id %in% c(test$device_id, train$device_id), ]
write.csv(count_by_period, "features/count_by_period.csv", row.names=FALSE)

## Apps

app_events = merge(app_events, events[, list(event_id, device_id)], by="event_id", all.x=TRUE)

apps_per_event = app_events[, list(installed=.N, active=sum(is_active)),
                            by=c("device_id", "event_id")]
apps_per_event_feature = apps_per_event[, list(max_apps_per_event = max(installed),
                                               min_apps_per_event = min(installed),
                                               avg_apps_per_event = mean(installed),
                                               max_active_per_event = max(active),
                                               min_active_per_event = min(active),
                                               avg_active_per_event = mean(active),
                                               avg_active_pct = mean(active/installed)),
                                        by="device_id"]
apps_per_event_feature = rbindlist(list(train[!device_id %in% apps_per_event_feature$device_id,
                                              list(device_id)],
                                        apps_per_event_feature), fill=TRUE)
apps_per_event_feature = rbindlist(list(test[!device_id %in% apps_per_event_feature$device_id,
                                             list(device_id)],
                                        apps_per_event_feature), fill=TRUE)
apps_per_event_feature = apps_per_event_feature[device_id %in% c(test$device_id, train$device_id), ]
write.csv(apps_per_event_feature, "features/apps_per_event.csv", row.names=FALSE)

installed_app_cnts = app_events[, .N, by=c("app_id", "device_id")]
installed_app_cnts = merge(installed_app_cnts, app_labels, by="app_id", allow.cartesian=TRUE)
installed_app_cnts = installed_app_cnts[, list(app_type_cnt=sum(N)), by=c("device_id", "label_id")]
installed_app_cnts = merge(installed_app_cnts, label_categories, by="label_id")
installed_app_cnts = installed_app_cnts[, list(app_type_cnt=sum(app_type_cnt)), by=c("device_id", "category")]
write.csv(installed_app_cnts, "features/installed_app_category_counts.csv", row.names=FALSE)

active_app_cnts = app_events[is_active == 1, .N, by=c("app_id", "device_id")]
active_app_cnts = merge(active_app_cnts, app_labels, by="app_id", allow.cartesian=TRUE)
active_app_cnts = active_app_cnts[, list(app_type_cnt=sum(N)), by=c("device_id", "label_id")]
active_app_cnts = merge(active_app_cnts, label_categories, by="label_id")
active_app_cnts = active_app_cnts[, list(app_type_cnt=sum(app_type_cnt)), by=c("device_id", "category")]
write.csv(active_app_cnts, "features/active_app_category_counts.csv", row.names=FALSE)

# Type of app: boolean, count, proportion of total by device_id