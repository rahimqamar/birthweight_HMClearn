#data frame contains the following columns:
  
#low      indicator of birth weight less than 2.5 kg.
#age      mother's age in years.
#lwt      mother's weight in pounds at last menstrual period.
#race     mother's race (1 = white, 2 = black, 3 = other).
#smoke    smoking status during pregnancy.
#ptl      number of previous premature labours.
#ht       history of hypertension.
#ui       presence of uterine irritability.
#ftv      number of physician visits during the first trimester.
#bwt      birth weight in grams.


library(hmclearn)
library(data.table)
library(gridExtra)
library(dplyr)
library(ggplot2)                       
library(ggcorrplot)
library(pROC)
library(caret)
library(Matrix)
library(glmnet)
library(coda)
library(bayesplot)

library(tidyverse)

library(stats)
library(class)
library(kknn)
library(rpart)  #decision tree classifier
library(randomForest)
library(xgboost)
library(lightgbm)
library(reticulate)  #CatBoostClassifier



library(boot)
library(MLmetrics)  # for F1, Precision, Recall (or use yardstick)
library(reshape2)


 birthwt2 = MASS::birthwt # data
 birthwt3 = as.data.frame(birthwt2, stringsAsFactors=FALSE)
 summary(birthwt2)
 head(birthwt2)
 
 
 #===========================================================================
 evaluate_model_performance = function(pred_probs, pred_labels, true_labels, MethodType) {
   library(caret)
   library(pROC)
   
   # Convert labels to factors with 0 and 1 as levels
   pred_factor = factor(pred_labels, levels = c(0, 1))
   true_factor = factor(true_labels, levels = c(0, 1))
   
   # Confusion matrix
   conf_matrix = confusionMatrix(pred_factor, true_factor, positive = "1")
   
   # Metrics from confusion matrix
   accuracy  = conf_matrix$overall["Accuracy"]
   recall    = conf_matrix$byClass["Sensitivity"]
   precision = conf_matrix$byClass["Precision"]
   f1        = conf_matrix$byClass["F1"]
   
   # AUC from predicted probabilities
   roc_obj = roc(response = true_labels, predictor = pred_probs)
   auc_val = auc(roc_obj)
   
   # Print results
   
   cat("Accuracy :", round(accuracy, 4), "\n")
   cat("Recall   :", round(recall, 4), "\n")
   cat("Precision:", round(precision, 4), "\n")
   cat("F1 Score :", round(f1, 4), "\n")
   cat("AUC      :", round(auc_val, 4), "\n")
   
   # Return all metrics as a list
   return(list(
     MType = MethodType,
     Accuracy = accuracy,
     Recall = recall,
     Precision = precision,
     F1 = f1,
     AUC = auc_val
   ))
 }
 
 
 #===========================================================================
 
 
 
 status_NA=is.na(birthwt2)
 birthwt2 = na.omit(birthwt2)
 
 # imbalance data set check
 
 tab = table(birthwt3)
 prop = prop.table(tab)

  if(min(prop) < 0.1) {
   cat("\nWarning: Data is imbalanced. Minority class proportion <", min(prop), "\n")
 } else {
   cat("\nData is relatively balanced.\n")
 }

 # to declare a column as categorical
 # categorical columns will be set as factor
 # if n is the number of values in a column and r are the number
 # occurrence for each value, then r < 0.05*n for a column to be declared as categorical
 
 
 plots = lapply(names(birthwt3), function(colnames){

    count_flds=birthwt3%>%count(.data[[colnames]])#(.data[[i]])
    if(nrow(count_flds) < nrow(birthwt3)*0.05){
      col = birthwt3 [[colnames]]
      data_display=ggplot(birthwt3,aes(x = col)) + geom_histogram() + labs(x = colnames)
      }
    else {
      col = birthwt3 [[colnames]]
      
      data_display=ggplot(birthwt3,aes(x = col)) + geom_density() + labs(x = colnames)
      }
 return(data_display)
 }
 )
 grid.arrange(grobs = plots, ncol = 2, nrow = 5)
 #      correlation analysis
 
 cor_matrix = cor(birthwt3, use = "complete.obs")
 
 
 #correlation plot
 ggcorrplot(cor_matrix, method = "circle", type = "lower",
            lab = TRUE, lab_size = 3, 
            colors = c("blue", "white", "red"),
            title = "Correlation Heatmap",
            ggtheme = theme_minimal())
 
 
 
# setting problem easier for understanding
 birthwt3$race2 = factor(birthwt3$race, labels = c("white", "black", "other"))
 # setting to a simpler problem, either previous premature labour or not
 birthwt3$ptd = ifelse(birthwt3$ptl > 0, 1, 0)
 # setting all values in ftv data >= 2 to 2+ because majority of the values are 0,1 and 2
 birthwt3$ftv2 = factor(ifelse(birthwt3$ftv > 2, 2, birthwt3$ftv), labels = c("0", "1", "2+"))
 
 #======== splitting data into train and test data ==========
 train_index = createDataPartition(birthwt3$low, p = 0.7, list = FALSE)
 train_df = birthwt3[train_index, ]
 test_df  = birthwt3[-train_index, ]
 #==================
 # scaling continous data
 num_cols = c("age", "lwt", "bwt")
 # Scale training numeric columns, storing as matrix to preserve attributes
 scaled_train = scale(train_df[num_cols])
 
 # Extract centering and scaling vectors from scaled_train attributes
 train_means = attr(scaled_train, "scaled:center")
 train_sds = attr(scaled_train, "scaled:scale")
 
 # Replace numeric columns in train_df with scaled values
 train_df[num_cols] = scaled_train
 
 # Scale test numeric columns using training means and sds
 test_df[num_cols] = scale(test_df[num_cols], center = train_means, scale = train_sds)
 summary(train_df)
 # building model
 
 X = model.matrix(low ~ age + lwt + race2 + smoke + ptd + ht + ui + ftv2, data = train_df)
 y = train_df$low
 
 # setting baseline results for HMC -------
 X_test = model.matrix(low ~ age + lwt + race2 + smoke + ptd + ht + ui + ftv2, data = test_df)
 
 X_train = model.matrix(low ~ age + lwt + race2 + smoke + ptd + ht + ui + ftv2, data=train_df)
 y_train = train_df$low
 

 y_new = test_df$low  # binary target
 
 #  ---- HMC learn ----- **************************************************
 logistic_posterior = function(theta, y, X, sig2beta=2.5**2) { k = length(theta)
 beta_param = as.numeric(theta)
 onev = rep(1, length(y))
 ll_bin = t(beta_param) %*% t(X) %*% (y - 1) -
   t(onev) %*% log(1 + exp(-X %*% beta_param))
 result = ll_bin - 1/2* t(beta_param) %*%
   beta_param / sig2beta
 return(result)
 }
 
 
 g_logistic_posterior = function(theta, y, X, sig2beta=2.5**2) { n = length(y)
 k = length(theta)
 beta_param = as.numeric(theta)
 result = t(X) %*% ( y - 1 + exp(-X %*% beta_param) /
                        (1 + exp(-X %*% beta_param))) -beta_param/sig2beta
 return(result)
 }
 
 
 
 N = 4000  # total iterations
 warmup = 2000  # warmup iterations, half of total
 continuous_ind = c(FALSE, TRUE, TRUE, rep(FALSE, 8))
 eps_vals = ifelse(continuous_ind, 1e-3, 5e-2)

 set.seed(143)
 fm2_hmc = hmc(N, theta.init = rep(0, 11),
                   epsilon = eps_vals, L = 10,
                   logPOSTERIOR = logistic_posterior,
                   glogPOSTERIOR = g_logistic_posterior,
                   param = list(y = y, X = X),
                   varnames = colnames(X),
                   chains = 2, parallel = FALSE)
# removing warmup samples
 posterior_beta = fm2_hmc$theta  # List of chains' samples (each chain is a list of samples)
 
 
   fm2_hmc$accept / N 
 # Extract posterior draws from both chains
 
 #====== diagnostics
 chain1 = as.numeric(unlist(fm2_hmc$theta[[1]]))
 chain2 = as.numeric(unlist(fm2_hmc$theta[[2]]))
 # Combine chains into mcmc.list for diagnostics
 mcmc_chains = mcmc.list(mcmc(chain1), mcmc(chain2))
 traceplot(mcmc_chains)
 
 # Effective Sample Size (ESS)
 ess_vals = effectiveSize(mcmc_chains)
 print(ess_vals)
 # Gelman-Rubin R-hat diagnostic
 rhat_vals <- gelman.diag(mcmc_chains)
 print(rhat_vals)
 
 # Acceptance rate
 accept_rate <- mean(fm2_hmc$accept/N)
 cat("Acceptance rate:", accept_rate, "\n")
 
 # Check if any divergent transitions recorded if available (depends on hmclearn output)
 if (!is.null(fm2_hmc$divergent)) {
   num_divergent <- sum(fm2_hmc$divergent)
   cat("Number of divergent transitions:", num_divergent, "\n")
 }
 
 #========= end of diagnostics=====
 
   X_new = model.matrix(low ~ age + lwt + race2 + smoke +
                       ptd + ht + ui + ftv2,
                     data = test_df)
   
   y_new = test_df$low  # binary target
   
   posterior_beta = fm2_hmc$theta  # Matrix of sampled betas

   combined_list = c(posterior_beta[[1]], posterior_beta[[2]])  # Combine samples from chains
   posterior_beta_matrix = do.call(rbind, combined_list)      # Convert list of vectors to matrix
   posterior_beta_matrix = posterior_beta_matrix[-(1:warmup), ]
   
   X_new = as.matrix(X_new)  # you already confirmed this
   linear_preds = X_new %*% t(posterior_beta_matrix)  
   probs_all = 1 / (1 + exp(-linear_preds))
   mean_probs = rowMeans(probs_all)
   

   predicted_class = ifelse(mean_probs > 0.4, 1, 0)
   #====
   accuracy = mean(predicted_class == y_new)
   cat("Test accuracy:", accuracy, "\n")

   
   MethodType = "Prediction using MClearn"
   
   
   result = evaluate_model_performance(mean_probs, predicted_class, y_new, MethodType)

   # for precision-recall curve plotting
   if (1) 
   {
   thresholds <- seq(0, 1, by = 0.01)
   # Initialize vectors to store precision and recall
   precision <- numeric(length(thresholds))
   recall <- numeric(length(thresholds))
   
   for (i in seq_along(thresholds)) {
     thresh <- thresholds[i]
     predicted_class <- ifelse(mean_probs > thresh, 1, 0)
     
     tp <- sum(predicted_class == 1 & y_new == 1)
     fp <- sum(predicted_class == 1 & y_new == 0)
     fn <- sum(predicted_class == 0 & y_new == 1)
     
     precision[i] <- ifelse(tp + fp == 0, 1, tp / (tp + fp))
     recall[i] <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
   }
   # Create a data frame for plotting
   pr_data <- data.frame(threshold = thresholds, precision = precision, recall = recall)
   
   # Plot Precision-Recall curve
   ggplot(pr_data, aes(x = recall, y = precision)) +
     geom_line(color = "blue") +
     labs(title = "Precision-Recall Curve",
          x = "Recall",
          y = "Precision") +
     theme_minimal()
   }
   
   
   
   # to do list, important point and explaining
   # precision recall
   # conditions
   # Rshiny
   # add a proper comparison
   
   
   
   
   
 