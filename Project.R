library(tseries)
library(forecast)
library(ggplot)

# Load the data
data <- read.csv("Case_study.csv")
y <- data$V82

# Step 4: Split the data
train_data <- y[1:100]
test_data <- y[101:200]

plot.ts(train_data, main="Time Series Plot of Yt", ylab="Value", col="blue")