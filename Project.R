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
results <- data.frame()

for(p in 1:4){
  for(q in 1:4){
    
    model <- Arima(y, order=c(p,1,q))
    
    results <- rbind(results,
                     data.frame(p=p,
                                q=q,
                                AIC=AIC(model),
                                BIC=BIC(model)))
  }
}

print(results)


results[order(results$AIC), ]
results[order(results$BIC), ]

summary(model)

#Step 3: Diagnostic tests with in-sample data

model1 <- Arima(y, order=c(2,1,2))
model2 <- Arima(y, order=c(3,1,2))
model3 <- Arima(y, order=c(4,1,1))
res <- residuals(model3)

Box.test(res, lag = 10, type = "Ljung-Box")




