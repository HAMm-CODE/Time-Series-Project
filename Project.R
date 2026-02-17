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

#Analysis of (p,q). I use the first differencing order to analyse p and q
stationary_series <- diff(train_data, differences = 1)
par(mfrow=c(1,2))
acf(stationary_series, main="ACF for q")
pacf(stationary_series, main="PACF for p")

#step 2: Estimation and selection of ARIMA Models
results <- data.frame()

for(p in 1:4){
  for(q in 1:4){
    
    model <- Arima(train_data, order=c(p,1,q))
    
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
#Do the tests for all the 3 best models
model1 <- Arima(y, order=c(2,1,2))
model2 <- Arima(y, order=c(3,1,2))
model3 <- Arima(y, order=c(4,1,1))
res <- residuals(model3)

Box.test(res, lag = 10, type = "Ljung-Box")

par(mfrow = c(1, 2))

res <- residuals(model3)

acf(res)
pacf(res)

#histogram
par(mfrow = c(1, 3))
res <- residuals(model3)
hist(res, probability = TRUE)
lines(density(res))

#qq plot
res <- residuals(model1)
qqnorm(res)
qqline(res, col = "red")

#Shapiro test
res <- residuals(model3)
shapiro.test(res)

#plot for best model
par(mfrow = c(1, 1))
plot(train_data, main="Original vs Fitted Model")
lines(fitted(model1), col="red")

#Step 4: Forecast with out-of-sample data






