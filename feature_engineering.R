###################################################################################################
## Load the libraries and data
###################################################################################################

library(data.table)
library(Matrix)
library(bit64)

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
event_counts = event_counts[device_id %in% train$device_id | device_id %in% test$device_id, ]
write.csv(event_counts, file="features/event_counts.csv", row.names=FALSE)

# geo location
avg_position = events[, list(avg_latitude=mean(latitude), avg_longitude=mean(longitude)),
                      by = device_id]
avg_position = rbind(data.table(device_id=train[!(has_event), device_id], avg_latitude=NA, avg_longitude=NA),
                     avg_position)
avg_position = rbind(data.table(device_id=test[!(has_event), device_id], avg_latitude=NA, avg_longitude=NA),
                     avg_position)
avg_position = avg_position[device_id %in% train$device_id | device_id %in% test$device_id, ]
write.csv(avg_position, file="features/avg_position.csv", row.names=FALSE)

sd_position = events[, list(sd_latitude=sd(latitude), sd_longitude=sd(longitude)),
                      by = device_id]
sd_position = rbind(data.table(device_id=train[!(has_event), device_id], sd_latitude=NA, sd_longitude=NA),
                     sd_position)
sd_position = rbind(data.table(device_id=test[!(has_event), device_id], sd_latitude=NA, sd_longitude=NA),
                     sd_position)
sd_position = sd_position[device_id %in% train$device_id | device_id %in% test$device_id, ]
write.csv(sd_position, file="features/sd_position.csv", row.names=FALSE)

# time
events[, timestamp := as.POSIXct(timestamp)]
qplot(events[, as.POSIXct(as.character(timestamp, "%H:%M:%S"), format="%H:%M:%S")])

## Apps

# Quantity of app events
# Apps active on events
# Time of usage (may vary by app)
# Type of app: boolean, count, proportion of total by device_id
# Average time (or maybe bucket times according to pre-work, work, post work, ...)