# Amazon-fine-food-review-analysis<br>

In this project we will perform Exploratory data analysis, data preprocessing, Feature engineering, model building etc on amazon fine food review and in each and every steps of this project, lots of several sub-steps are included. Built various model like KNN, Naive Bayes, Logistic Regressin, Decision Tree, All types of Clustering, XGBoost etc with many featurization tecnique like bow, tfidf, word2vec, average word2vec, tfidf word2vec etc and also performed hyperparameter tuning for each and every model and plotted various plot for checking model stability, convergence of hyperparameter value, underfitting, overfitting etc.
 <p align="center">
      <img src="https://i.imgur.com/bApB5TM.jpg" />
 </p>

# Data Source and Information
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.
  * https://www.kaggle.com/snap/amazon-fine-food-reviews
  * Number of reviews: 568,454
  * Number of users: 256,059
  * Number of products: 74,258
  * Timespan: Oct 1999 - Oct 2012
  * Number of Attributes/Columns in data: 10 
  
# Attribute Information
  * Id
  * ProductId - unique identifier for the product
  * UserId - unqiue identifier for the user
  * ProfileName
  * HelpfulnessNumerator - number of users who found the review helpful
  * HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
  * Score - rating between 1 and 5
  * Time - timestamp for the review
  * Summary - brief summary of the review
  * Text - text of the review
  
# Objective
We have given a review and we have to identify whether a review is positive(rating of 4 or 5) or negative(rating of 1 or 2).

# KNN Algorithm
  * Applied 4 featurization technique on text data called bow, tfidf, average word2vec and tfidf word2vec.
  * Tuned hyperparameter k using knn brute force algorithm with scoring parameter was accuracy and plotted coresponding plot.
    <p align="center">
      <img src="https://i.imgur.com/jMAuGF4.png" />
    </p>
  <b>Conclusions</b><br>
  * As in "knn with tfidf" when k = 49 the accuracy is quite good than other models. In this model, train_error and test_error is low.
  * As we know when a model performs good on training data but poor performence on unseen data(test data)i.e. its dependent on training data only, tends towards overfits and when a model perform poor performence on training data and good performence on test data i.e. it fails to learn relationship in training data tends towards underfit. We need to balance between both i.e. reduce training error and reduce error between training and testing error.
  * Another concept bias vs variance is also related with underfitting and overfitting. when a model has high bias and low variance tend towards underfitting and its reverse- high variance and low bias called overfitting and we balanced using cross-validataion. As it is shown in below table where first three models have low trainig error and test error. But the accuracy it low which we can boost using some techniques.
  * There are lot more things to write here but for now that's all.
# Naive Bayes
 * Featurize the text data using the previous technique.
 * Tuned hyperparameter to get best alpha and plotted the graph at each iteration.
 * Obtained top 10 +Ve and -Ve features.
   <p align="center">
      <img src="https://i.imgur.com/ITymB7U.png" />
   </p>
 <b>Conclusions</b><br>
 * Naive bayes are good at text classification task like spam filtering, sentimental analysis, RS etc.
 * As we know when a model performs good on training data but poor performence on unseen data(test data)i.e. its dependent on training data only, tends to overfits and when a model perform poor performence on training data and good performence on test data i.e. it fails to learn relationship in training data tends to underfit. We need to balance between both i.e. reduce training error and balance error between both training and testing which is balanced in this case.
 * Another concept bias vs variance is also related with underfitting and overfitting. when a model has high bias and low variance tend to underfitting and its reverse- high variance and low bias called overfitting and we balanced using cross-validataion. As it is shown in below table where both models have low trainig error and test error.
 * overall, both of the models are performing well on unseen data.
 * As we are not applying naive bayes on word2vec representation because it sometimes gives -ve value(i.e. if two word have 0 cosine similarity the word is completly orthogonal i.e. they are not related with each other. and 1 represents perfect relationship between word vector. whereas -ve similarity means they are perfect opposite relationship between word) and we know naive bayes assume that presence of a particular feature in a class is unrelated to presence of any other feature, which is most unlikely in real word. Although, it works well.
 * And from point # 5, features are dependent or there are relationship between features. So applying naive bayes on dependent feature does not make any sense.
 
# Logistic Regression 
 * Featurize text data using the classical technique.
 * Applied grid search and reandom search for all the featurization and find best hyperparameter lambda.
 * Plotted precision_reacall graph along with find f1-score, auc-score, precision score etc.
 * Find multi-collinearity using pertubation test.
 * Get feature importance for non-collinear features.
   <p align="center">
      <img src="https://i.imgur.com/IyAABTp.png" />
      <img src="https://i.imgur.com/RFZM8AD.png" />
   </p>
 <b>Conclusions</b><br>
  * We reduced training error and balance error between both training and testing. Although, cross-validataion do not completly remove underfitting or overfitting.
  * Bow and tfidf is working well whereas avg word2vec and tfidf w2v is like dumb model
  
# Support Vector Machine
 * As usual on each and every featurization applied grid search on two types of svm called rbf(radial basis function) kernel and linear kernel and find two hyperparameter c and alpha.
 * plotted word cloud of all the features.
   <p align="center">
      <img src="https://i.imgur.com/RhUs2Xl.png" />
      <img src="https://i.imgur.com/ddCaXYU.png" />
   </p>
 <b>Conclusions</b><br>
  * Applied 4 technique to vectorize text data and work with 2 version of svc linear and rbf. For linear svc using bow and tfidf, we took 100k data points and for kernel svc using same we took 25k datapoints as it is computationaly very expensive. For word2vec featurization we took 100k datapoints for both version on svc.
  * Did gridsearch cross validation and find hyperparameter c and gamma.
    We also calibrated using sigmoid method as both version of svc is giving probabilisitc output. Calibration is necessary when we get probabilities score but not for all model as we do not need it for logistic regression because the predicted probabilities is already calibrated. https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
  * Observe that svc worked really well with bow, tfidf representation of text data and did not perform well on word2vec representation of text data.
  * At the end of this cell, provided overall summery of model which will help to understand model performence completly in a single go.

# Decision Tree
 * Did Hyperparameter tuning on all the featurization and find the best depth of the tree.
 * plotted depth vs error plot, word cloud and visualize tree upto 2 node.
   <p align="center">
      <img src="https://i.imgur.com/MznnFLc.png" />
      <img src="https://i.imgur.com/vcnrNVA.png" />
      <img src="https://i.imgur.com/Sj7N1l3.png" />
      <img src="https://i.imgur.com/gqqtaSI.png" />
   </p>
 <b>Conclusions</b><br>
  * Decision tree often does not work well with high dimension data but for bow and tfidf works slightly good than w2v.
  * overall, None of the models are performing well on unseen data, when our performence measure is "roc_auc_score". Here, We are dealing with imbalanced data, hence we should also try f1-score, balanced accuracy etc
 
# XGBoost and RandomForest
 * For each text representation, tuned hyperparameter(max_depth and n_estimator) and calculated score for XGBOOSt and RF.
 * Random forest works well on bow and tfidf representations of text. Whereas, xgboost fails to perform better in all the four vectorization.
 * RF & XGBOOST have many parameters to tune but worked on only 2 parameter here, as we wanted to show it in heatmap.
 * Printed feature importance in a word cloud, so that it is easy to clearly identify in a single shot, which feature is most important.
 * Calculated roc_auc score because data was imbalanced and it works better than accuracy, in this situation
  <p align="center">
      <img src="https://i.imgur.com/mrLF52o.png" />
      <img src="https://i.imgur.com/HtxOGOm.png" />
      <img src="https://i.imgur.com/8Y6sZcy.png" />
  </p>
  
# K-means, Agglomerative, DBSCAN Clustering 
 * Used kmeans clustering with 50k datapoints and DBSCAN with 5k data-points on 4 set of techniques and for each featurization, choose k value and eps using elbow method respectively and also used agglomerative clustering with 5k data-points and just choose two different number of cluster(2 and 5) and plotted word cloud of all review, for all clustering technique.
 * Observe that in kmeans and agglomerative clustering, average word2vec perform slightely well. We are saying this just by looking at the cloud cluster because did not apply any technique to check performence of kmeans or agglomerative clustering. In DBSCAN we can say that it not good as it clustered everything in a single cluster and as the review given by user is not same type of review.
 * I have written comment in each section of each technique please go through comments.
  <p align="center">
      <img src="https://i.imgur.com/724ontz.png" />
      <img src="https://i.imgur.com/WyyBtk5.png" />
  </p>
 
# Truncated SVD
 * Used 100k data-points and applied tfidf on top of it to vectorize text data and then selected top 2k features based on idf score.
 * alculated co-occurence matrix to store count of how often a features occur together in a context and then used truncatedsvd.
 * Used k-means clustering to group similar features together.
 * Calculated cosine similarity to get which words are more similar to a given word. 
 <p align="center">
      <img src="https://i.imgur.com/HcQK0SO.png" />
      <img src="https://i.imgur.com/Gx4iINT.png" />
  </p>

 
