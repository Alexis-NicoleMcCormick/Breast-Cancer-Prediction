# Breast Cancer Prediction Project

library(tidyverse)
library(caret)
library(dplyr)
library(ggplot2)
library(dslabs)

options(digits=3)
data(brca)

#brca$y is a vector of sample classifications
#brca$x is a matrix of numeric features describing properties of the cell nuclei

#How many samples are in the data set? (Count the number of rows in brca$x)
dim(brca$x)[1]

569

#How many predictors are in the matrix? (Count the number of columns in brca$x)
dim(brca$x)[2]

30

#What proportion of the samples are malignant? (the avg malignant tumors)
mean(brca$y == "M")

0.373

#Which column number has the highest mean?

which.max(colMeans(brca$x))

area_worst: 24

#Which column has the lowest standard deviation
which.min(colSds(brca$x))

fractal_dim_se: 20

#Now we must scale or normalize our matrix. Use sweep() function 2 times to scale each column by subtracting the column means of brca$x (center the values), then divide by the column standard deviations of brca$x (scale).

x_centered<- sweep(brca$x, 2, colMeans(brca$x))
x_scaled<- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

#What is the standard deviation of the first column after scaling

sd(x_scaled[,1])

1

#After scaling what is the median value of the first column?

median(x_scaled[,1])

-0.215

#Calculate the distance between all samples using the scaled matrix

dist(x_scaled)
d_samples<- dist(x_scaled)

#what is the avg distance between the first sample, which is benign, and other benign samples?
dist_btob <- as.matrix (d_samples)[1, brca$y =="B"]
mean(dist_btob[2:length(dist_btob)])


#What is the average distance between the first sample and malignant samples?
dist_BtoM <- as.matrix(d_samples)[1, brca$y == "M"]
mean(dist_BtoM)

#Make a heatmap of the relationship between features using the scaled matrix
#Note that column names and row labels can be removed using labRow= NA and labCol= NA
#Note that the t() function is used to transpose the data(turn the rows into columns and the columns into rows)
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow =NA, labCol= NA)

#Perform hierarchial clustering on 30 features. Cut the tree into 5 groups
h<- hclust(d_features)
groups<- cutree (h, k=5)
split(names(groups), groups)

#Part 2
#Perform a principal component analysis (PCA) of the scaled matrix
pca<- prcomp(x_scaled)

# What proportion of variance is explained by the first principal component?
# How many principal components are required to explain at least 90% of the variance?
summary(pca)
0.443,
7

#Plot the first two principal components with color representing tumor type (benign/malignant)
data.frame(pca$x[,1:2], type = brca$y) %>%
ggplot(aes(PC1, PC2, color= type))+
geom_point()
#Notice how malignant tumors tend to have larger values of PC1 than benign tumors

#Make a box plot of the first 10 principal components grouped by tumor type
data.frame(type= brca$y, pca$x[, 1:10]) %>%
gather(key = "PC", value = "value", -type) %>%
ggplot(aes(PC, value, fill=type)) +
geom_boxplot()

#Which PCs are significantly different enough by tumor type that there is no overlap in the interquartile ranges (IQRs) for benign and malignant samples?
PC1

#Part 3
#Set the seed to 1, then create a data partition splitting brca$y and the scaled version of the brca$x matrix into a 20% test set and 80% train set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition (brca$y, times=1, p=0.2, list=FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

#Make sure the training and test sets have similar proportions of benign and malignant tumors

mean(train_y =="B")

mean(test_y == "B")

#The predict_kmeans() function defined below takes 2 arguments ( a matrix of observations "x", and a k-means object k) and assigns each row of x to a cluster from k

predict_kmeans <- function (x,k) {
	centers<- k$centers  #extract cluster centers
	distances<- sapply(1:nrow(x), function(i) {
		apply(centers, 1, function(y) dist(rbind(x[i,], y))) #calculate distance to cluster centers
	})
	max.col(-t(distances)) #select cluster with min distance to center
}

#set the seed to 3. Perform k-means clustering on the training set with 2 centers and assign the output to k. Then use the predict_kmeans() function to make predictions on the test set.
set.seed(3, sample.kind = "Rounding")
k <- kmeans(train_x, centers = 2)
kmeans_preds <- ifelse(predict_kmeans(test_x, k)== 1, "B", "M")

#What is the overall accuracy?
mean(kmeans_preds == test_y)

#What proportion of benign tumors are correctly identified?

sensitivity(factor(kmeans_preds), test_y, positive = "B")

#What proportion of malignant tumors are correctly identified?

sensitivity(factor(kmeans_preds), test_y, positive = "M")

#Fit a logistic regression model on the training set with caret::train() using all predictors. Make predictions on the test set
train_glm <- train(train_x, train_y, method ="glm")
glm_preds <- predict(train_glm, test_x)

#What is the accuracy of the logistic regression model on the test set?
mean(glm_pred == test_y)

#Train an LDA model on the training set. Make predictions on the test set using the LDA model
train_lda <- train(train_x, train_y, method ="lda")
lda_preds <- predict(train_lda, test_x)

#What is the accuracy of the LDA model on the test set?
mean(lda_preds == test_y)

#Train a QDA model on the training set. Make predictions on the test set using the QDA model
train_qda  <- train(train_x, train_y, method= "qda")
qda_preds <- predict(train_qda, test_x)

#What is the accuracy of the qda model on the test set?
mean(qda_preds == test_y)

install.packages("gam")
library(gam)

#Set the seed to 5 and fit a loess model on the training set with the caret package. Use the default tuning grid and make predictions on the test set
set.seed(5, sample.kind = "Rounding")
train_loess <- train(train_x, train_y, method = "gamLoess")
loess_preds <- predict(train_loess, test_x)

#What is the accuracy of the loess model on the test set
mean(loess_preds == test_y)

#Set the seed to 7 and train a knn model on the train set. Try odd values of k from 3 to 21. Use the final model to make predictions on the test set
set.seed(7, sample.kind = "Rounding")
tuning <- data.frame(k = seq (3, 21, 2))
train_knn <- train(train_x, train_y, method = "knn", tuneGrid = tuning)

#What is the final value of k used in the model?
train_knn$bestTune

#What is the accuracy of the knn model on the test set
knn_preds <- predict(train_knn, test_x)
mean(knn_preds == test_y)

#set the seed to 9, then train a random forest model on the training set using the caret package. Test mtry values of c(3, 5, 7, 9). Use the argument importance = TRUE so that feature importance can be extracted. Make predictions on the test set

set.seed(9, sample.kind = "Rounding")
tuning <- data.frame(mtry = c(3, 5, 7, 9))
train_rf <- train(train_x, train_y, method = "rf", tuneGried = tuning, importance = TRUE)

#What value of mtry gives the highest accuracy

train_rf$bestTune

#What is the accuracy of the random forest model on the test set?

rf_preds <- predict(train_rf, test_x)
mean(rf_preds == test_y)

#What is the most important variable in the random forest model? Considering the top 10 most imporatnt variables in the random forest model, which set of features is nost important for determining tumor type?

varImp(train_rf)


#create an ensemble using the predictions from the 7 models. Use the ensemble to generate a majority prediction of the tumor type (if most models suggest malignant, predict malignant)

ensemble <- cbind(glm =glm_preds == "B", lda_preds == "B", qda = qda_preds == "B", loess = loess_preds == "B", rf= rf_preds == "B", knn= knn_preds == "B", kmeans = kmeans_preds == "B")

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "B", "M")

#What is the accuracy of the ensemble prediction
mean(ensemble_preds == test_y)

#Make a table of the accuracies of the 7 models and the accuracy of the ensemble model.
#Which model has the highest accuracy?

models <- C("K means", "Logistic regression", "LDA", "QDA", "Loess", "K nearest neeighbors", "Random forest", "Ensemble")
accuracy <- c(mean(kmeans_preds == test_y),
mean(glm_preds == test_y), mean(lda_preds == test_y),
mean(qda_preds == test_y),
mean(loess_preds == test_y),
mean(knn_preds == test_y),
mean(rf_preds == test_y),
mean(ensemble_preds == test_y))

data.frame(Model= models, Accuracy = accuracy)

