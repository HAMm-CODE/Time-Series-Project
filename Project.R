
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

#Sort the results to take in ascending order to take the smallest AIC and BIC
results[order(results$AIC), ]
results[order(results$BIC), ]

model1 <- Arima(train_data, order=c(3,1,1))
model2 <- Arima(train_data, order=c(3,1,2))
model3 <- Arima(train_data, order=c(4,1,1))
#using summary to get the estimates of the 3 models
summary(model1)
summary(model2)
summary(model3)


#Step 3: Diagnostic tests with in-sample data
# Ljung-Box test for first 10 lags
Box.test(residuals(model1), lag = 10, type = "Ljung-Box")
Box.test(residuals(model2), lag = 10, type = "Ljung-Box")
Box.test(residuals(model3), lag = 10, type = "Ljung-Box")

#Now we plot ACF and PACF of residuals
par(mfrow = c(1,2))

acf(residuals(model1), main="ACF Residuals (3,1,1)")
pacf(residuals(model1), main="PACF Residuals (3,1,1)")

acf(residuals(model2), main="ACF Residuals (3,1,2)")
pacf(residuals(model2), main="PACF Residuals (3,1,2)")

acf(residuals(model3), main="ACF Residuals (4,1,1)")
pacf(residuals(model3), main="PACF Residuals (4,1,1)")


#histogram
par(mfrow = c(1,3))

hist(residuals(model1), probability=TRUE, main="Histogram (3,1,1)")
lines(density(residuals(model1)))

hist(residuals(model2), probability=TRUE, main="Histogram (3,1,2)")
lines(density(residuals(model2)))

hist(residuals(model3), probability=TRUE, main="Histogram (4,1,1)")
lines(density(residuals(model3)))

#qq plots
par(mfrow = c(1,3))

qqnorm(residuals(model1), main="QQ (3,1,1)")
qqline(residuals(model1), col="red")

qqnorm(residuals(model2), main="QQ (3,1,2)")
qqline(residuals(model2), col="red")

qqnorm(residuals(model3), main="QQ (4,1,1)")
qqline(residuals(model3), col="red")

#Shapiro-Wilk Normality Test
shapiro.test(residuals(model1))
shapiro.test(residuals(model2))
shapiro.test(residuals(model3))


#plot for best model
par(mfrow=c(1,1))

plot(train_data, main="Original vs Fitted ARIMA(3,1,1)")
lines(fitted(model1), col="red")
legend("topleft",
       legend=c("Original","Fitted"),
       col=c("black","red"),
       lty=2)

#Step 4: Forecast with out-of-sample data

best_model <- Arima(train_data, order=c(3,1,1))

forecast_10 <- forecast(best_model, h=10, level=95)
plot(forecast_10, main="10-Step Forecast (ARIMA(3,1,1))")


forecast_25 <- forecast(best_model, h=25, level=95)
plot(forecast_25, main="25-Step Forecast (ARIMA(3,1,1))")


forecast_100 <- forecast(best_model, h=100, level=95)
plot(forecast_100, main="100-Step Forecast (ARIMA(3,1,1))")

test_data <- y[101:200]

#MSE for h = 10
pred_10 <- forecast(best_model, h=10)$mean
actual_10 <- test_data[1:10]

mse_10 <- mean((actual_10 - pred_10)^2)
mse_10

#MSE for h = 25
pred_25 <- forecast(best_model, h=25)$mean
actual_25 <- test_data[1:25]

mse_25 <- mean((actual_25 - pred_25)^2)
mse_25

#MSE for h = 100
pred_100 <- forecast(best_model, h=100)$mean
actual_100 <- test_data[1:100]

mse_100 <- mean((actual_100 - pred_100)^2)
mse_100





