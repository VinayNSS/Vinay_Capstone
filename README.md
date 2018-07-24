# Vinay_Capstone
Analysis of Glassdoor Reviews

Prediction of Glassdoor Rating & Sentiment Based on Review Text

ABSTRACT 
In this project, the goal is to scrape employee reviews for a company from Glassdoor website and predict the rating of the reviews based on machine learning and Natural Language Processing techniques. The goal is to predict star rating and also sentiment rating to see which one can be predicted with better accuracy.

Introduction:
Majority of the online reviews - product or company or opinions - are written in free-text format. It is often useful to have a measure which summarizes the content of the review. One such measure can be sentiment which expresses the polarity (positive/negative) of the review. However, a more granular classification such as review stars, would be more accurate depiction of user sentiment or product/service rating. This project suggests an approach which involves a combination of topic modeling and sentiment analysis to achieve this objective and thereby help predict the rating stars.

Data Sources:
Glassdoor website was scraped to obtain employee reviews for a company.
The technique used was to provide the initial URL for the company employee review page to the program which scraped the page, extracted the tags, retrieved the elements from the page that were intended and then formed the URL for the next page. All the review data is then gathered in a dataframe and saved off to a csv file. There were over 26000 review collected for Amazon for this project. Main libraries used were BeautifulSoup & urllib.

Data Exploration:

Rating counts:


Frequency counts for Pros & Cons:
Word Tokenization, lemmatization, removing stopwords gives us Top 100 words in Pros and Cons for the Reviews
Pros:

Pros Wordcloud:


Cons:



Ratings Prediction

What is the baseline model?
Baseline Rating: 4
            precision    recall  f1-score   support

        1.0       0.00      0.00      0.00      2112
        2.0       0.00      0.00      0.00      2430
        3.0       0.00      0.00      0.00      4847
        4.0       0.31      1.00      0.47      7020
        5.0       0.00      0.00      0.00      6237

avg / total       0.10      0.31      0.15     22646

Created a feature vector out of review text using bag of words and vectorization so that we can classify the reviews into Ratings
Used CountVectorizer to convert the text collection into a matrix of token counts
Using train_test_split, created training and test dataset
Ran various classifiers to find Ratings
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')


             precision    recall  f1-score   support

        1.0       0.26      0.08      0.12       679
        2.0       0.16      0.02      0.04       752
        3.0       0.24      0.23      0.23      1412
        4.0       0.32      0.45      0.38      2081
        5.0       0.34      0.41      0.37      1870

avg / total       0.29      0.31      0.28      6794

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

             precision    recall  f1-score   support

        1.0       0.52      0.38      0.44       679
        2.0       0.24      0.05      0.08       752
        3.0       0.31      0.24      0.27      1412
        4.0       0.38      0.64      0.48      2081
        5.0       0.52      0.45      0.48      1870

avg / total       0.40      0.41      0.39      6794

LogisticRegression(C=1.0, class_weight=None,dual=False,fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
             precision    recall  f1-score   support

        1.0       0.52      0.35      0.42       679
        2.0       0.31      0.17      0.22       752
        3.0       0.33      0.30      0.31      1412
        4.0       0.39      0.49      0.44      2081
        5.0       0.47      0.52      0.49      1870

avg / total       0.40      0.41      0.40      6794

RandomForestClassifier(bootstrap=True,class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=2,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

             precision    recall  f1-score   support

        1.0       0.60      0.18      0.28       679
        2.0       0.40      0.03      0.05       752
        3.0       0.31      0.21      0.25      1412
        4.0       0.36      0.61      0.45      2081
        5.0       0.45      0.50      0.47      1870

avg / total       0.40      0.39      0.35      6794

AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=100, random_state=None)

             precision    recall  f1-score   support

        1.0       0.45      0.33      0.38       679
        2.0       0.25      0.08      0.12       752
        3.0       0.29      0.18      0.23      1412
        4.0       0.36      0.60      0.45      2081
        5.0       0.46      0.40      0.43      1870

avg / total       0.37      0.38      0.35      6794

Summary of Results for different models:


Improving Prediction by using Sentiment.
Rating > 3.5 is categorized as positive
   		precision    recall  f1-score   support

          0       0.71      0.59      0.64      2843
          1       0.74      0.82      0.78      3951

avg / total       0.72      0.73      0.72      6794

The prediction is much better now that the Rating is not that granular !!

Conclusion
The task of mining opinion from a review was successful. The model to predict if the review is positive or negative is 72% accurate. However, when we try to assign a scale to the review from 1 to 5, it is not that accurate. It is because words are difficult to quantify into a rating on a numeric scale. Each person is different is their choice of words in review but may use a rating quite different than overall positive or negative tone. More research is needed to increase the accuracy of predictions based on reviews. The application of the techniques has wide applications in text mining, information retrieval & sentiment analysis on the world wide web. At the least, sentiment analysis of Glassdoor reviews would give HR personnel & Companies a tool to measure & manage employee satisfaction.
