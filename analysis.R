#=============================================================
# ICO FUNDRAISING SUCCESS PREDICTION
# Description: Predicts ICO fundraising campaign success using
#              KNN, Decision Tree, Random Forest and SVM.
#              Includes data cleaning, feature engineering,
#              10-fold cross-validation and model evaluation.
# packages: lubridate, ggplot2, corrplot, corrgram, VIM,
#           quantmod, cowplot, patchwork, xtable, class,
#           kernlab, C50, randomForest, pROC, caret, dplyr
#=============================================================

library(lubridate)
library(ggplot2)
library(corrplot)
library(corrgram)
library(VIM)
library(quantmod)
library(cowplot)
library(patchwork)
library(xtable)
library(class)        
library(kernlab)    
library(C50)          
library(randomForest)
library(pROC)
library(caret)
library(dplyr)

#=============================================================
# 1. DATA UNDERSTANDING
#=============================================================


# clear workspace
rm(list = ls())

# load data - update path as needed
ico <- read.csv('LUBS5990M_courseworkData_202526.csv', header = TRUE)
head(ico)
str(ico)

# summary statistics
summary(ico)

# class distribution of binary variables
lapply(ico[c("goal", "whitelist", "kyc", "mvp", "bonus", "ERC20")], table)


#-------------------------------------------------------------
# Visualisations
#-------------------------------------------------------------
bin_colours <- c("#A81E1E", "#4A964B")

# class distribution of target variable
barplot(table(ico$goal),
        main = "Class Distribution of ICO Outcome",
        xlab = "Outcome of ICO", 
        ylab = "Count", 
        ylim = c(0, 4000),
        names.arg = c("Failure", "Success"), 
        col = bin_colours)
# proportion of each class
table(ico$goal) / nrow(ico) * 100
# uneven target class distribution 
# a naive classifier would get 65.84% accuracy by predicting all cases fail

# binary variable distributions (shows inconsistent encoding)
par(mfrow = c(2,3), oma = c(1, 0, 0, 0)) 
for (v in c("goal", "whitelist", "kyc", "bonus", "mvp", "ERC20")) {
  barplot(table(ico[[v]]), col = "#5AA9B0", las = 3, main = v, ylab = "Count")
}

# continuous variable distributions
par(mfrow = c(2, 3), mar=c(5, 4, 4, 2)) 
# price log histogram
hist(log1p(ico$price_usd), main = "log(price_usd + 1)",
     xlab = "Log Token Price (log(USD + 1))",
     col = "#5AA9B0", breaks = 30, ylab = "Count",
     cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)
# rating histogram
hist(ico$rating, 
     main = "rating", 
     xlab = "Rating", 
     col = "#5AA0B0", breaks = 30, ylab = "Count",
     cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)
# team size histogram
hist(ico$teamsize, main = "teamsize",
     xlab = "Number of Team Members",
     col = "#5AA0B0", breaks = 30, ylab = "Count",
     cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)
# tokens for sale log histogram
hist(log1p(ico$token_for_sale),
     main = "log(token_for_sale + 1)",
     xlab = "Log Number of Tokens for Sale",
     col = "#5AA0B0", ylab = "Count",
     cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)
# distributed_in_ico log histogram
hist(log1p(ico$distributed_in_ico), 
     main = "log(distributed_in_ico + 1)",
     xlab = "Log Percentage of Tokens Offered", 
     col = "#5AA0B0", breaks = 30, ylab = "Count",
     cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)

# vastly missing continuous variables
# sold tokens histogram
hist(log1p(ico$sold_tokens),
     main = "Distribution of Number of Tokens Sold",
     xlab = "log(sold_tokens + 1)", col = "#5AA0B0",
     breaks = 30, cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)
# pre sale price histogram
hist(log1p(ico$pre_ico_price_usd), main = "Distribution of Presale Token Price (log scale)",
     xlab = "log(pre_ico_price_usd + 1)",
     col = "#5AA0B0", breaks = 30, cex.main = 1.2, cex.axis = 1.2, cex.lab = 1.2)

# summary statistics for continuous variables
summary(ico[c("price_usd", "rating", "teamsize", "token_for_sale", 
              "distributed_in_ico", "sold_tokens", "pre_ico_price_usd")])


# inspect missing data 
sum(!complete.cases(ico)) # Rows with missing data = 5813
sum(!complete.cases(ico))/nrow(ico)*100
colSums(is.na(ico)) / nrow(ico) * 100 # percentage NAs by column
aggr(ico,
     prop=TRUE,
     cex.axis = 1.6,
     cex.lab = 1.4,
     col = c("#5AA0B0", "#A81E1E"),
     oma = c(12,5,3,2)) # missing data distribution

#=============================================================
# 2. DATA PREPARATION
#=============================================================

#-------------------------------------------------------------
# 2.1 Handling Incorrect Values and Outliers
#-------------------------------------------------------------

ico_copy <- ico

# binary variable standardisation 
# inspect unique values to confirm inconsistent encoding
lapply(ico_copy[c("goal", "whitelist", "kyc", "mvp", "bonus", "ERC20")], unique)

# define positive and negative string variants
yes <- c("1", "1.0", "yes", "y", "true", "available", "present", "t")
no <- c("0", "0.0", "no", "n", "false", "f", "")

# make binarise function
binarise <- function(x) {
  x <- trimws(tolower(x)) # trim white space and make lowercase
  ifelse(x %in% yes, 1, 
         ifelse(x %in% no, 0, NA))
}

# make list of binary columns
binary_cols <- c("goal", "whitelist", "kyc", "mvp", "bonus", "ERC20")

# apply function
ico_copy[binary_cols] <- lapply(ico_copy[binary_cols], binarise)

# check binarisation by looking at value tables
lapply(ico_copy[binary_cols], table, useNA = "always")


# date parsing
ico_copy$ico_start     <- dmy(ico_copy$ico_start)
ico_copy$ico_end       <- dmy(ico_copy$ico_end)
ico_copy$pre_ico_start <- dmy(ico_copy$pre_ico_start)
ico_copy$pre_ico_end   <- dmy(ico_copy$pre_ico_end)


# create sale and pre-sale duration variables
ico_copy$duration <- as.numeric(ico_copy$ico_end - ico_copy$ico_start)
ico_copy$pre_duration <- as.numeric(ico_copy$pre_ico_end - ico_copy$pre_ico_start)

summary(ico_copy$duration)
summary(ico_copy$pre_duration)

# drop rows where the sale or presale duration is negative
ico_copy <- ico_copy[is.na(ico_copy$duration) 
                     | !is.na(ico_copy$duration) & ico_copy$duration >= 0, ] # 5828 -> 5774 rows
ico_copy <- ico_copy[is.na(ico_copy$pre_duration) 
                     | !is.na(ico_copy$pre_duration) & ico_copy$pre_duration >= 0, ] # 5774 -> 5768 rows

# inspect long duration rows - manual cross-referencing confirmed date entry errors
# e.g. row 10: ico_end = 2020-12-31, actual end = 2019-03-31 (source: https://icoholder.com/en/satt-21400)
ico_copy[ico_copy$duration > 365 & !is.na(ico_copy$duration), ]
# remove campaigns with duration > 365 days as confirmed data entry errors
ico_copy <- ico_copy[is.na(ico_copy$duration)     | ico_copy$duration <= 365, ]     # -> 5693
ico_copy <- ico_copy[is.na(ico_copy$pre_duration) | ico_copy$pre_duration <= 365, ] # -> 5685


# drop price outliers - rows where the price is in the 99th percentile (> 227.91)
quantile(ico_copy$price_usd, 0.99, na.rm = TRUE)
ico_copy <- ico_copy[!is.na(ico_copy$price_usd) & ico_copy$price_usd <= quantile(ico_copy$price_usd, 0.99, na.rm = TRUE) 
                     | is.na(ico_copy$price_usd), ] # 5685 -> 5632 rows

# distributed_in_ico values <= 1 assumed entered as proportions,convert to percentages (e.g. 0.02 -> 2%)
ico_copy$distributed_in_ico <- ifelse(ico_copy$distributed_in_ico < 1,
                                      ico_copy$distributed_in_ico * 100,
                                      ico_copy$distributed_in_ico)
# drop rows where distributed_in_ico is over 100% - impossible
ico_copy <- ico_copy[is.na(ico_copy$distributed_in_ico) 
                     | !is.na(ico_copy$distributed_in_ico) & ico_copy$distributed_in_ico <= 100, ] # 5632 -> 5433 rows

# drop tokens for sale outliers
summary(ico_copy$token_for_sale)
quantile(ico$token_for_sale, 0.99, na.rm = TRUE)
ico_copy <- ico_copy[!is.na(ico_copy$token_for_sale) & ico_copy$token_for_sale <= quantile(ico_copy$token_for_sale, 0.99, na.rm = TRUE) 
                     | is.na(ico_copy$token_for_sale), ] # 5433 -> 5389 rows
#-------------------------------------------------------------
# 2.2 Feature Engineering
#-------------------------------------------------------------
# note: feature engineering before missing value handling to ensure
# source columns (dates, URLs, country) are available for derivation

# create presale indicator
ico_copy$had_pre_ico <- ifelse(is.na(ico_copy$pre_ico_start)&is.na(ico_copy$pre_ico_end), 0, 1)

# set pre-sale duration to 0 if had_pre_ico == 0 
ico_copy$pre_duration <- ifelse(ico_copy$had_pre_ico == 0, 0, ico_copy$pre_duration)

# drop presale start and end as over half NA, need numerical variables, and captured by pre-duration
ico_copy$pre_ico_start <- NULL
ico_copy$pre_ico_end <- NULL


# binary variables for presence of URLs
ico_copy$has_whitepaper <- ifelse(is.na(ico_copy$link_white_paper) | ico_copy$link_white_paper == "", 0, 1)
ico_copy$has_linkedin <- ifelse(is.na(ico_copy$linkedin_link) | ico_copy$linkedin_link == "", 0, 1)
ico_copy$has_github     <- ifelse(is.na(ico_copy$github_link) | ico_copy$github_link == "", 0, 1)
ico_copy$has_website    <- ifelse(is.na(ico_copy$website) | ico_copy$website == "", 0, 1)
# drop url variables
ico_copy$link_white_paper <- NULL
ico_copy$linkedin_link <- NULL
ico_copy$github_link <- NULL
ico_copy$website <- NULL


# create binary variable for whether in US, Singapore or UK (most prevalent)
sort(table(ico_copy$country), decreasing = TRUE)
ico_copy$is_US_Sing_UK <- ifelse(tolower(ico_copy$country) %in% c("usa", "uk", "singapore"), 1, 0)
# drop country variable
ico_copy$country <- NULL
table(ico_copy$is_US_Sing_UK)/ nrow(ico_copy)*100

# log transform token_for_sale, drop original column
ico_copy$log_token_for_sale <- log1p(ico_copy$token_for_sale)
ico_copy$token_for_sale <- NULL

# adding bitcoin price variable
summary(ico_copy$ico_start) # min = 2014-01-18, max = 2021-02-22
# download Bitcoin price history for range of ico sale dates
getSymbols("BTC-USD", src = "yahoo", from = "2014-01-18", to = "2021-02-22")
btc <- data.frame(date = index(`BTC-USD`), 
                  btc_price = as.numeric(`BTC-USD`[, "BTC-USD.Close"]))
# convert to date dmy format 
btc$date <- as.Date(btc$date)
# create variable for bitcoin price at start date by merging
ico_copy <- merge(ico_copy, btc, 
                  by.x = "ico_start", 
                  by.y = "date", 
                  all.x = TRUE)

dev.off()
# visualise Bitcoin price over sample period
plot(btc_price ~ date, data = btc, type = "l",
     main = "Bitcoin Closing Price 2014-2021",
     xlab = "Date", ylab = "Price (USD)")

# bitcoin price distribution histogram - log scale
hist(log1p(ico_copy$btc_price), col = "#5AA0B0", main = "Distribution of Bitcoin Price 2014-2021 (log scale)", xlab = "log(btc_price + 1)")

# drop sale date variables as captured by duration and need numerical variables
ico_copy$ico_start <- NULL
ico_copy$ico_end <- NULL

#-------------------------------------------------------------
# 2.3 Handling Missing Values
#-------------------------------------------------------------

# inspect missing data 
sum(!complete.cases(ico_copy)) # Rows with missing data = 5424
colSums(is.na(ico_copy)) / nrow(ico_copy) * 100 # percentage NAs by column
aggr(ico_copy,
     prop=TRUE,
     cex.axis = 1.2,
     cex.lab = 1.2,
     col = c("#5AA0B0", "#A81E1E"),
     oma = c(10,5,3,2)) # missing data distribution

# make copy to handle missing values so can compare distributions
ico_clean <- ico_copy

# drop columns with >70% missingness - insufficient for modelling
ico_clean$sold_tokens      <- NULL  # 97% missing
ico_clean$pre_ico_price_usd <- NULL # 75% missing

# missingness in duration and btc_price is caused by missingness in ico_start and ico_end
# both variables are important predictors with heavily skewed distributions
# making imputation unreliable - listwise deletion applied
ico_clean <- ico_clean[!is.na(ico_clean$duration), ] # 5389 -> 4682 rows

# pre-duration only 0.08% missing so list-wise deletion
ico_clean <- ico_clean[!is.na(ico_clean$pre_duration),] # 4682 -> 4678 rows

# price_usd: <10% missing, listwise deletion avoids imputation of core variable
ico_clean <- ico_clean[!is.na(ico_clean$price_usd),] # 4678 -> 4475 rows

# remaining variables: missingness appears independent (MCAR)
# missingness too high for list-wise deletion, median imputation selected
# median preferred over mean due to skewed distributions
ico_clean$rating <- ifelse(is.na(ico_clean$rating), median(ico_clean$rating, na.rm = TRUE), ico_clean$rating)
ico_clean$log_token_for_sale <- ifelse(is.na(ico_clean$log_token_for_sale), median(ico_clean$log_token_for_sale, na.rm = TRUE), ico_clean$log_token_for_sale)
ico_clean$teamsize <- ifelse(is.na(ico_clean$teamsize), median(ico_clean$teamsize, na.rm = TRUE), ico_clean$teamsize)
ico_clean$distributed_in_ico <- ifelse(is.na(ico_clean$distributed_in_ico),
                                 median(ico_clean$distributed_in_ico, na.rm = TRUE),
                                 ico_clean$distributed_in_ico)

# check all missings dealt with 
sum(!complete.cases(ico_clean)) # no missing values 

# now compare distributions before and after handling missing data
# continuous variables
cont_vars = c("price_usd", "distributed_in_ico", "rating", "teamsize",
              "duration", "pre_duration", "log_token_for_sale", "btc_price")

before_stats <- sapply(cont_vars, function(v) {
  x <- ico_copy[[v]]
  c(n = sum(!is.na(x)),
    mean = round(mean(x, na.rm = TRUE), 2),
    median = round(median(x, na.rm = TRUE), 2),
    sd = round(sd(x, na.rm = TRUE), 2))
}, simplify = "matrix")

after_stats <- sapply(cont_vars, function(v) {
  x <- ico_clean[[v]]
  c(n = sum(!is.na(x)),
    mean = round(mean(x, na.rm = TRUE), 2),
    median = round(median(x, na.rm = TRUE), 2),
    sd = round(sd(x, na.rm = TRUE), 2))
}, simplify = "matrix")

stats_df <- data.frame(
  Variable      = cont_vars,
  N_Before      = before_stats["n", ],
  N_After       = after_stats["n", ],
  Mean_Before   = before_stats["mean", ],
  Mean_After    = after_stats["mean", ],
  Median_Before = before_stats["median", ],
  Median_After  = after_stats["median", ],
  SD_Before     = before_stats["sd", ],
  SD_After      = after_stats["sd", ]
)

stats_df

# before/after histograms for imputed variables
# add status column to each dataset
ico_copy$status  <- "Before"
ico_clean$status <- "After"

clean_vars <- colnames(ico_clean)
clean_vars
# combine datasets
combined <- rbind(
  ico_copy[, c(clean_vars, "status")],
  ico_clean[, c(clean_vars, "status")]
)

# set factor order so Before plots behind After
combined$status <- factor(combined$status, levels = c("Before", "After"))

# rating
p1 <- ggplot(combined, aes(x = rating, fill = status)) +
  geom_histogram(bins = 20, position = "identity", alpha = 0.8) +
  scale_fill_manual(values = c("Before" = "#A81E1E", "After" = "#5AA0B0"),
                    name = "Missing Value\nHandling Status") +
  labs(title = "Distribution of rating",
       x = "Rating", y = "Count") +
  theme_minimal()

# teamsize
p2 <- ggplot(combined, aes(x = teamsize, fill = status)) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.8) +
  scale_fill_manual(values = c("Before" = "#A81E1E", "After" = "#5AA0B0"),
                    name = "Missing Value\nHandling Status") +
  labs(title = "Distribution of teamsize",
       x = "Team Size", y = "Count") +
  theme_minimal()

# distributed_in_ico
p3 <- ggplot(combined, aes(x = distributed_in_ico, fill = status)) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.8) +
  scale_fill_manual(values = c("Before" = "#A81E1E", "After" = "#5AA0B0"),
                    name = "Missing Value\nHandling Status") +
  labs(title = "Distribution of distributed_in_ico",
       x = "% Tokens Distributed to Investors", y = "Count") +
  theme_minimal()

# log_token_for_sale 
p4 <- ggplot(combined, aes(x = log_token_for_sale, fill = status)) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.8) +
  scale_fill_manual(values = c("Before" = "#A81E1E", "After" = "#5AA0B0"),
                    name = "Missing Value\nHandling Status") +
  labs(title = "Distribution of log_token_for_sale",
       x = "log(Tokens for Sale + 1)", y = "Count") +
  theme_minimal() 


# combine plots
(p1 + p2) /
  (p3 + p4)

# remove status column before modelling
ico_clean$status <- NULL

nrow(ico_clean)      # 4475 rows
ncol(ico_clean) - 1  # 19 predictor variables

#-------------------------------------------------------------
# 2.4 Relationships Between Variables
#-------------------------------------------------------------

# correlation matrix of continuous predictors 
cor(ico_clean[cont_vars])
corrgram(ico_clean[cont_vars], upper.panel = panel.cor, cex.labels = 1)

# boxplots comparing distributions of continuous variables and target variable
par(mfrow = c(2,4), mar = c(5, 5, 3, 2))
# duration
boxplot(duration ~ goal, data = ico_clean, col = bin_colours, main = "duration", ylab = "duration (days)")
# pre-duration
boxplot(pre_duration ~ goal, data = ico_clean, col = bin_colours, main = "pre_duration", ylab = "pre-duration (days)")
# teamsize
boxplot(teamsize ~ goal, data = ico_clean, col = bin_colours, main = "teamsize")
# rating
boxplot(rating ~ goal, data = ico_clean, col = bin_colours, main = "rating")
# distributed in ico
boxplot(distributed_in_ico ~ goal, data = ico_clean, col = bin_colours, main = "disstributed_in_ico", ylab = "distributed_in_ico (%)")
# price
boxplot(log1p(price_usd) ~ goal, data = ico_clean, col = bin_colours, main = "log(price_usd + 1)", ylab = "log(price_usd + 1) (log(USD + 1))")
# bitcoin price
boxplot(log1p(btc_price) ~ goal, data = ico_clean, col = bin_colours, main = "log(btc_price + 1)", ylab = "log(btc_price + 1)  (log(USD + 1))")
# log_token_for_sale
boxplot(log_token_for_sale ~ goal, data = ico_clean, col = bin_colours, main = "log_token_for_sale", ylab = "log_token_for_sale")


# stacked barplots of binary variables vs goal
bin_vars = c("whitelist", "kyc", "bonus", "mvp", "ERC20", "had_pre_ico", "is_US_Sing_UK", "has_whitepaper", "has_linkedin", "has_github", "has_website")

par(mfrow = c(2,6))  

for (col in bin_vars) {
  barplot(prop.table(table(ico_clean$goal, ico_clean[[col]]), margin = 2), beside = FALSE,
          col = bin_colours,
          main = col,
          xlab = col, ylab = "Proportion")
}
plot.new()  # create an empty plot space
legend("left",
       title = "goal",
       legend = c("0    ", "1    "),
       fill = bin_colours,
       cex = 1.5)

# check bonus class distribution by ico outcome
table(ico_clean$goal[ico_clean$bonus == 1])/nrow(ico_clean[ico_clean$bonus == 1,])*100
table(ico_clean$goal[ico_clean$bonus == 0])/nrow(ico_clean[ico_clean$bonus == 0,])*100
# check bonus class distribution
table(ico_clean$bonus)/nrow(ico_clean)*100
# bonus very unbalanced so difference in ico outcome by class may not be reliable

#=============================================================
# 3. MODELLING
#=============================================================

#-------------------------------------------------------------
# 3.1 Data Splitting
#-------------------------------------------------------------

model_ico <- ico_clean
model_ico$goal <- as.factor(model_ico$goal)
TRAIN_RATIO <- 0.8
FIXED_SEED <- 123

# Create train-test split
set.seed(FIXED_SEED)
train_size <- floor(TRAIN_RATIO * nrow(model_ico))
train_indices <- sample(nrow(model_ico), train_size)
ico_train <- model_ico[train_indices, ]
ico_test <- model_ico[-train_indices, ]

# check goal distribution of train and test set
table(ico_train$goal)/nrow(ico_train)*100 # 60.37% : 39.62%
table(ico_test$goal)/nrow(ico_test)*100 # 61.68% : 38.32%

# create folds for stratified 10-fold cross validation
set.seed(FIXED_SEED)
folds <- createFolds(ico_train$goal, k = 10)

# verify consistent class distribution across folds
sapply(folds, function(f) {
  table(ico_train[f, ]$goal) / length(f) * 100
})

#-----------------------------------------------------------------------------
# Decision Tree
#-----------------------------------------------------------------------------
# 10-fold CV to tune hyperparameters (number of boost trials), evaluating mean AUC
trials <- 1:20
dt_cv_results <- data.frame()

for (t in trials) {

  fold_aucs <- c()
  for (i in 1:10) {
    val_idx <- folds[[i]]
    cv_train <- ico_train[-val_idx, ]
    cv_val <- ico_train[val_idx, ]
    
    dt_model <- C5.0(goal ~ ., data = cv_train, trials = t)
    dt_preds <- predict(dt_model, cv_val)
    dt_probs <- predict(dt_model, cv_val, type = "prob")[, 2]
    
    dt_roc <- roc(cv_val$goal, dt_probs)
    fold_aucs[i] <- as.numeric(auc(dt_roc))
  }
  
  dt_cv_results <- bind_rows(dt_cv_results, data.frame(trials = t,
                                             mean_auc = mean(fold_aucs),
                                             sd_auc = sd(fold_aucs)))
}

dev.off()

plot(dt_cv_results$trials, dt_cv_results$mean_auc,
     type = "b",
     col = "#0E7BB2",
     pch = 20,
     xlab = "Number of Trials",
     ylab = "Score",
     main = "AUC vs Number of Trials")


best_trials_auc <- dt_cv_results$trials[which.max(dt_cv_results$mean_auc)]
best_trials_auc # 16
summary(dt_cv_results$mean_auc)


# train final model with 16 trials
set.seed(FIXED_SEED)
dt_model <-  C5.0(goal ~ ., data = ico_train, trials = 16)

# evaluate on test set 
dt_preds <- predict(dt_model, ico_test)
dt_probs <- predict(dt_model, ico_test, type = "prob")[, 2]
dt_cm <- confusionMatrix(dt_preds, ico_test$goal, positive = "1")
dt_cm$byClass["F1"] # f1 = 0.520

dt_roc <- roc(ico_test$goal, dt_probs)
dt_auc <- auc(dt_roc) 
dt_auc # auc = 0.724

#-------------------------------------------------------------
# 3.3 Random Forest
#-------------------------------------------------------------
# 10-fold CV to tune number of trees, evaluating mean AUC
ntrees <- c(50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
rf_cv_results <- data.frame()

for (n in ntrees) {
  fold_aucs <- c()
  
  for (i in 1:10) {
    val_idx <- folds[[i]]
    cv_train <- ico_train[-val_idx, ]
    cv_val <- ico_train[val_idx, ]
    
    rf_model <- randomForest(goal ~ ., data = cv_train, 
                 ntree = n)
    rf_preds <- predict(rf_model, cv_val)
    rf_probs <- predict(rf_model, cv_val, type = "prob")[, 2]
    
    rf_roc <- roc(cv_val$goal, rf_probs)
    fold_aucs[i] <- as.numeric(auc(rf_roc))
    
  }
  
  rf_cv_results <- bind_rows(rf_cv_results, data.frame(ntrees = n,
                                                   mean_auc = mean(fold_aucs),
                                                   sd_auc = sd(fold_aucs)))
}

plot(rf_cv_results$ntrees, rf_cv_results$mean_auc,
     type = "b",
     col = "#0E7BB2",
     pch = 20,
     xlab = "Number of Trees",
     ylab = "Score",
     main = "AUC vs Number of Trees")

best_trees_auc <- rf_cv_results$ntrees[which.max(rf_cv_results$mean_auc)]
best_trees_auc # 500
summary(rf_cv_results$mean_auc)


# train model with ntree = 500 on whole training set
set.seed(FIXED_SEED)
rf_model <- randomForest(goal ~ ., data = ico_train, 
                         ntree = 500, importance = TRUE)
rf_preds <- predict(rf_model, ico_test)
rf_probs <- predict(rf_model, ico_test, type = "prob")[, 2]

# confusion matrix
rf_cm <- confusionMatrix(rf_preds, ico_test$goal, positive = "1")
rf_cm$byClass["F1"]# F1 = 0.524

rf_roc <- roc(ico_test$goal, rf_probs)
rf_auc <- as.numeric(auc(rf_roc))
rf_auc  # auc = 0.732

# variable importance plot
varImpPlot(rf_model, main = "Random Forest Variable Importance")

#-------------------------------------------------------------
# 3.4 Support Vector Machine
#-------------------------------------------------------------
# tune cost and kernel via 10-fold CV, evaluating mean AUC

costs <- c(0.01, 0.1, 1, 10)
kernels <- c("vanilladot", "rbfdot", "polydot")

svm_cv_results <- data.frame()

for (c in costs) {
  for (k in kernels) {
    fold_aucs <- c()
    
    for (i in 1:10) {
      val_idx <- folds[[i]]
      cv_train <- ico_train[-val_idx, ]
      cv_val <- ico_train[val_idx, ]
      
      svm_model <- ksvm(goal ~ .,
                        data = cv_train,
                        kernel = k,
                        C = c,
                        scaled = TRUE,
                        prob.model = TRUE)
      
      svm_preds <-  predict(svm_model, cv_val)
      svm_probs <- predict(svm_model, cv_val, 
                           type = "probabilities")[, 2]
      
      
      svm_roc <- roc(cv_val$goal, svm_probs)
      fold_aucs[i] <- as.numeric(auc(svm_roc))
    }
    
    svm_cv_results <- bind_rows(svm_cv_results, data.frame(cost = c,
                                                       kernel = k,
                                                       mean_auc = mean(fold_aucs),
                                                       sd_auc = sd(fold_aucs)))
  }
}

# select cost and kernel which achieved best mean auc in cv
best_svm_row_auc <- svm_cv_results[which.max(svm_cv_results$mean_auc), ]
best_svm_row_auc # cost = 1, kernel = rbfdot

# train model on entire training set with cost = 1 and kernel = rbfdot
set.seed(FIXED_SEED)
svm_model <- ksvm(goal ~ .,
                  data = ico_train,
                  kernel = "rbfdot",
                  C = 1,
                  scaled = TRUE,
                  prob.model = TRUE)

svm_preds <-  predict(svm_model, ico_test)
svm_probs <- predict(svm_model, ico_test, 
                            type = "probabilities")[, 2]

svm_cm <- confusionMatrix(svm_preds, ico_test$goal, positive = "1")
svm_cm$byClass["F1"] # F1 = 0.549

svm_roc <- roc(ico_test$goal, svm_probs)
svm_auc <- as.numeric(auc(svm_roc))
svm_auc # AUC = 0.721


#-------------------------------------------------------------
# 3.5 K-Nearest Neighbour
#-------------------------------------------------------------

# Z-score normalisation
# make knn train and test sets with only predictor variables to normalise
ico_train_knn <- select(ico_train, -goal)
ico_test_knn <- select(ico_test, -goal)

# compute training set statistics
train_means <- sapply(ico_train_knn, mean)
train_sds   <- sapply(ico_train_knn, sd)

# apply z-score normalisation to predictors using the training mean and sds
ico_train_knn <- as.data.frame(scale(ico_train_knn))

# Apply the training mean and sd to normalise the test set
ico_test_knn <- as.data.frame(scale(ico_test_knn,
                                    center = train_means, 
                                    scale  = train_sds))

# set goal columns to goal from original train and test sets
ico_train_knn$goal <- ico_train$goal
ico_test_knn$goal <- ico_test$goal

# k likely best aorund square root of train size
sqrt(nrow(ico_train_knn)) # 60

# 10-fold CV to tune k-value hyperparameter (odd so no vote ties)
k_values <- c(1, 5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121)
knn_cv_results <- data.frame()

for (k in k_values) {
  fold_aucs <- c()
  for (i in 1:10) {
    val_idx <- folds[[i]]
    cv_train <- ico_train_knn[-val_idx, ]
    cv_val <- ico_train_knn[val_idx, ]
    
    knn_preds <- knn(train = cv_train[, colnames(cv_train) != "goal"],
                     test  = cv_val[,  colnames(cv_val)  != "goal"],
                     cl    = cv_train$goal,
                     k     = k,
                     prob = TRUE)
    
    # extract the vote proportions
    knn_probs_raw <- attr(knn_preds, "prob")
    # get probabilities for evaluation
    knn_probs <- ifelse(knn_preds == 1, knn_probs_raw, 1 - knn_probs_raw)
    
    knn_roc <- roc(cv_val$goal, knn_probs)
    fold_aucs[i] <- as.numeric(auc(knn_roc))
  }
  
  knn_cv_results <- rbind(knn_cv_results, data.frame(k = k,
                                                   mean_auc = mean(fold_aucs),
                                                   sd_auc = sd(fold_aucs)))
}
dev.off()

plot(knn_cv_results$k, knn_cv_results$mean_auc,
     type = "b",
     col = "#0E7BB2",
     pch = 20,
     xlab = "K Value",
     ylab = "Score",
     main = "AUC vs K Value")

best_k_auc <- knn_cv_results$k[which.max(knn_cv_results$mean_auc)]
best_k_auc # 111

# graph has an elbow at around 61
# train model with k = 61 on whole training set, balance performance and complexity
knn_preds <- knn(train = ico_train_knn[, colnames(ico_train_knn) != "goal"], # remove goal from training as specified in cl parameter
                 test  = ico_test_knn[,  colnames(ico_test_knn)  != "goal"], # remove goal from test set as knn function doesn't
                 cl    = ico_train_knn$goal,
                 k     = 61,
                 prob  = TRUE)

# extract the vote proportions
knn_probs_raw <- attr(knn_preds, "prob")
# get probabilities for evaluation
knn_probs <- ifelse(knn_preds == 1, knn_probs_raw, 1 - knn_probs_raw)

# confusion matrix
knn_cm <- confusionMatrix(knn_preds, ico_test_knn$goal, positive = "1")
knn_cm$byClass["F1"]  # f1 = 0.506

knn_roc <- roc(ico_test$goal, knn_probs)
knn_auc<- as.numeric(auc(knn_roc))
knn_auc # AUC = 0.719

#=============================================================
# 4. EVALUATION
#=============================================================

# performance metrics dataframe
extract_metrics <- function(cm) {
  c(
    Accuracy  <- cm$overall["Accuracy"],
    Precision <- cm$byClass["Precision"],
    Recall    <- cm$byClass["Recall"],
    F1        <- cm$byClass["F1"]
  )
}

# build the results table
results_df <- rbind(
  DecisionTree = extract_metrics(dt_cm),
  RandomForest = extract_metrics(rf_cm),
  SVM          = extract_metrics(svm_cm),
  KNN          = extract_metrics(knn_cm)
)

# convert to dataframe
results_df <- as.data.frame(results_df)
results_df


# ROC curves and AUC
# Plot settings
par(mar = c(4, 4, 4, 4))  # Adjust margins

# combined plot
plot(dt_roc,  col = "#6a994e",   main = "ROC Curves - All Models")
plot(rf_roc,  col = "#177e89",    add = TRUE)
plot(svm_roc, col = "#ee964b", add = TRUE)
plot(knn_roc,  col = "#A35276",  add = TRUE)

legend("bottomright",
       legend = c(paste("Decision Tree:", round(dt_auc, 3)),
                  paste("Random Forest:", round(rf_auc, 3)),
                  paste("SVM:", round(svm_auc, 3)),
                  paste("KNN:", round(knn_auc, 3))),
col = c("#6a994e", "#177e89", "#ee964b", "#A35276"), lwd = 2)









