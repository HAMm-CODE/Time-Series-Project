library(tseries)
library(forecast)
library(ggplot2)


data <- read.csv("Case_study.csv")
y <- data$V82


train_data <- y[1:100]
test_data <- y[101:200]

plot.ts(train_data, main="Time Series Plot of Yt", ylab="Value", col="blue")

acf(train_data, main="ACF of Yt")

acf(diff(train_data, differences = 1), main="ACF of ∇Yt")

acf(diff(train_data, differences = 2), main="ACF of ∇2Yt")

adf_test <- adf.test(train_data)
print(adf_test)

adf_test1 <- adf.test(diff(train_data, differences = 1))
print(adf_test1)

adf_test2 <- adf.test(diff(train_data, differences = 2))
print(adf_test2)

stationary_series <- diff(train_data, differences = 1)
par(mfrow=c(1,2))
acf(stationary_series, main="ACF for q")
pacf(stationary_series, main="PACF for p")

#step 2




