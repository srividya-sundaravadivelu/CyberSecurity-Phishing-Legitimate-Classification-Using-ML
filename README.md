<h1 align="center">Cyber Security Phishing Legitimate Classification Using Machine Learning</h1>
<h2>Problem Statement:</h2>
Book-My-Show will enable the ads on their website, but they are also very cautious about their user privacy and information who visit their website. Some ads URL could contain a malicious link that can trick any recipient and lead to a malware installation, freezing the system as part of a ransomware attack or revealing sensitive information. Book-My-Show now wants to analyze that whether the particular URL is prone to phishing (malicious) or not. Using Machine Learning techniques, we need to predict if a given URL is “Phishing” or “Legitimate”.
<h2>Importing the libraries:</h2>
We will be using several Python libraries and frameworks specific to Machine Learning. We have imported the libraries for pandas, numpy, scipy, and scikit-learn which are used for data processing and Machine Learning. We also use matplotlib and seaborn for exploratory data analysis and visualizations.
![image](https://user-images.githubusercontent.com/89622446/168639470-5949483d-d482-423b-aa2a-abfa253a9274.png)
<h2>Data Retrieval:</h2>
The input dataset contains an 11k sample corresponding to the 11k URL. Each sample contains 32 features that give a different and unique description of URL ranging from -1,0,1.
 1: Phishing
 0: Suspicious
 1: Legitimate
The sample could be either legitimate or phishing. We downloaded the dataset using pandas into a data frame.
<h2>Data Exploration</h2>
Exploratory data analysis is done to explore and understand the given Cyber Security dataset. We do feature selection to include only the relevant features. Features that are least correlated to dependent variable are discarded. Also, independent variables that are highly correlated among themselves are also discarded as they denote redundancy. 
<h4>1.	Number of Samples</h4>
The dataframe has 11055 rows and 32 columns. "Result" column is the y variable (Dependent variable). Since it is labelled data, we are dealing with a supervised learning problem.
<h4>2.	Check for null values </h4>
There are no null values in any of the columns. All the columns are of type integer.
<h4>3.	Get the unique values </h4>
All the columns have only 2 or all of the values : -1, 0 and 1.
-1: Phishing 0: Suspicious 1: Legitimate
The "Result" column has 2 possible outcomes -1 and 1 where -1 is Phishing and 1 is Legitimate.
The "Result" is the dependent variable which we must predict. This is the "y" variable. Since the "Result" has 2 classes, this is a Binary classification problem.
<h4>4.	Heat map </h4>
From the heatmap, its evident that "popUpWindow" and "Favicon" are having strong correlations, since the color is almost white(closer to 1.0). This suggests data redundancy. One of these columns must be removed.
<h4>5.	Remove highly correlated features </h4>
We need to find the independent variables which are highly correlated among themselves. This means there is redundant data. We need to drop these redundant columns. Lets set the threshold as 0.75 and select the columns that are having correlation greater than 0.75. These columns must be dropped from the dataframe. 
The above code returned five columns which are highly correlated among themselves. We will be dropping these columns from the dataframe.
<h4>6.	Remove weakly correlated features to target </h4>
"Result" is having strong correlation with "SSLfinal_State" and    URL_of_Anchor. 'Redirect', 'RightClick', 'Iframe', 'Favicon' are having a weak correlation with "Result". We dropped the columns which have correlation less than or equal to 0.03. These columns have very little significance to the Result column.
After dropping the columns with correlation <= 0.03, we are left with 21 columns.
<h4>7.	Check for class imbalance </h4>
We can see that around 55 % of values belong to class 1 and 44 percentage of values belong to class -1. We don’t have the imbalanced class problem here.
<h4>8.	Histogram </h4>
Above histograms shows the data distribution of all the columns in the dataset. We can clearly    see that the unique values are -1,0 and 1.
<h4>9.	Count plot </h4>
<h4>10.	Insights from EDA </h4>
1. From the above graphs, we can clearly see that when SSLFinal_State is Legitimate, the "Result"   is mostly "Legitimate". When the SSLFinal_State is "Phishing", the "Result" is also mostly "Phishing". This means we can see a strong positive correlation between SSLFinalState and Result columns.

2. Also, when the URL_of_Anchor is "Legitimate", the "Result" is most of the time "Legitimate".    When the URL_of_Anchor is "Phishing", the "Result" is also "Phishing" most of the time. Hence there is a positive correlation between URL_of_Anchor and Result.

3. The web_traffic and Result seem to be having a positive correlation. When the web_traffic is "Legitimate", the "Result" is mostly "Legitimate".

<h2>Model Building</h2>
<h4>Model 1: SGD Classifier</h4>
                        
<h5>1.	Performance metrics for Train Data</h5>
Let’s use K Fold cross validation to evaluate the model. Here we divide the train data into 5 folds and reserve 1 for validation and 4 for training. We continue the steps till all the folds participate in both training and validation. By this way, we can validate the model more accurately and keep the test data untouched till the model is ready and finished training. Also, cross validation resolves the problem of over-fitting.

i.	Accuracy

We get around 91 % accuracy in all the 5 folds. But for a classification problem, accuracy is not a good measure. There are other measures like Confusion Matrix, Precision, Recall and F1 score.

ii.	Confusion Matrix 

There are 345 + 276 = 621 wrong predictions. There are 3054 + 4063 = 7117 right predictions. Accuracy is 7117/(7117 + 621) = 92%.

iii.	Precision, Recall and F1 Score

iv.	ROC Curve

The ROC curve is a plot of True Positive Rate (TPR) versus the False Positive Rate (FPR). 
The dotted line shows a very ineffective classifier, similar to random chance. 
Our SGD Classifier looks effective as its closer to the top left margin.

v.	AUC

The ROC curve can be used to generate the AUC. AUC is the total area under the ROC curve.  

<h5>2.	Performance metrics for Test Data</h5>
<i>(refer the code - Cyber_Security_Capstone (3).ipynb)</i>
 
<h4>Model 2: Logistic Regression</h4>
<h5>1.	Performance metrics for Train Data</h5>

i.	Accuracy 

ii.	Confusion Matrix 

iii.	Precision, Recall and F1 Score 

iv.	ROC Curve 

v.	AUC 

<h5>2.	Performance metrics for Test Data</h5>
 
Logistic Regression and SGD classifier are almost equal in terms of model efficiency, their accuracy approximately 92%.


<h4>Model 3: Random Forest Classifier</h4>
<h5>1.	Performance metrics for Train Data</h5>
i.	Grid Search CV
 
ii.	Accuracy
 
iii.	Confusion Matrix
 
iv.	Precision, Recall and F1 Score
 
v.	ROC Curve
 
vi.	AUC
 
<h5>2.	Performance metrics for Test Data</h5>
 
We can clearly see that Random Forest is the winner out of the 3 models with an accuracy of 97% , F1 score of 0.97 and AUC of 0.97.

<h5>3.	Feature Importance</h5>
 
We can see the feature importance for the random forest model. The top three features that contributed toward classifying an URL as "Phishing" or "Legitimate" are SSLFinal_State, URL_of_Anchor and web_traffic.

<h2>Conclusion</h2>
In this Project, we analyzed the cyber security dataset classifying an URL as "Phishing" or "Legitimate". We explored various features in the dataset by drawing graphs. We buit classification models like SGD Classifier, Logistic Regression and Random Forest. We did performance metrics for each of these models. We can conclude that the Random Forest model is the winner among the three models with an accuracy of 97%. The F1 score is 97% and the AUC is 0.97. This indicates that we have built an effective model which can classify the URLs correctly with very minimal misclassification.
