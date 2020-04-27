# Project 3: Using Natural Language Processing to Distinguish Among Reddit Posts

## Data Science Problem

For this project, we were tasked with finding two different subreddits, culling posts from each, and using Natual Language Processing (NLP) to see how well we can classify posts as belonging to one or other subreddit.

With the timing of this project coinciding with the 2020 NFL Draft, I selected the subreddits r/cowboys and r/eagles, two rival NFL franchises with dedicated fan bases \[I myself have been a Cowboys fan since I was six years old.\]

**PRIMARY DATA SCIENCE PROBLEM**

Build a machine learning classifier that maximizes the accuracy of predicting r/cowboys posts ***without using obvious keywords/tokens***, but with particular emphasis on maximizing the model's Sensitivity (in other words, trying to minimize the chances of falsely predicting a true Cowboys fan to be an Eagles fan).

**SECONDARY DATA SCIENCE PROBLEM**

Uncover and diagnose which words are more indicative of one or the other fan bases heading into the draft.

## Executive Summary

In order to execute this project, I collected 2000 reddit posts using the [Pushshift API](https://github.com/pushshift/api), scraping 1000 posts from the r/cowboys subreddit and 1000 posts from the r/eagles subreddit. Since I wanted to detect r/cowboys posts, I coded each subdataset as either a 1 (r/cowboys posts) or 0 (r/eagles posts), setting up a binary classification problem.

Because this project is about Natural Language Processing and not image evaluation, the two fields from the posts that I focused on for analysis were the "title" and "selftext" fields. Titles are typically like headlines and may contain anything from a few words to a couple of sentences. Selftext fields, when present, are more like paragraphs in their form. For the analysis, I concatenated the title and selftext fields into a single (sometimes lengthy) string. As a reference, about three-quarters of posts only consist of a title.

At this point, the two files were merged, shuffled, and randomly split into train and test data files for modeling, with 1400 cases selected for training the machine learning classifier and 600 posts used as a holdout sample to evaluate model performance. Stratification was used to ensure that each of these files contained 50% r/cowboys posts and 50% r/eagles posts.

The two datasets were cleaned by removing html and non-letter characters, converting all remaining characters to lowercase, and splitting the strings into "tokens", which are essentially any words or characters that are separated by spaces. These tokens were then lemmatized, which reduces multiple forms of the same word into a common base form. Stopwords were then removed in the final step prior to analysis. Stopwords are either words that have been consistently judged to provide little meaning in text analysis, or in this case, words that so obviously point to being from either subreddit as to make classification too easy. I wanted my machine learners to work hard!

The analysis stream consisted of running the data through either of two types of vectorizers in order to convert the raw text data into numeric variables suitable for modeling. Count Vectorization simply sums or counts tokens used within each post (so values are integer counts from zero to theoretically infinity), while Tfidf Vectorization converts tokens to relative ratios of how many times a word occurs in a given post relative to the number of times it occurs across all posts in the data set. After vectorization, the data were run through a series of classification models, including:

    - Logistic Regression Classifier
    - Multinomial Naive Bayes Classifier
    - Decision Tree Classifier
    - Bagged Decision Tree Classifier
    - Random Forests Classifier
    - Extra Trees Classifier
    - Gradient Boosting Classifier
    - Support Vector Machine Classifier

Model performance was judged against the baseline accuracy rate of 50%, as well as an "unfair" model where custom stopwords were not removed. Again, I was looking for a model that would be relatively high in accuracy, balance Sensitivity and Specificity (as shown by a high ROC AUC score), but lean towards maximizing Sensitivity.

**Results**

Somewhat surprisingly, even the "unfair" model didn't have an easy time of it, only correctly predicting 80% of the posts (which is still a substantial improvement over the naive 50% accuracy rate baseline). The remaining models achieved accuracy ranging between 56-70% on cross-validated data. The "winning" model included Count Vectorized data passed through a Logistic Regression classifier with L1 (LASSO) regularization, resulting in a solution with 67% cross-validated accuracy, 72% Sensitivity, and a ROC AUC score of 72%. These represent decent performance, but not world-class classification levels.


## Contents
**Code**<br>
[Technical Report](./code/technical-report.ipynb)

**Presentation**<br>
[NLP Machine Learning Classification](./presentation/NLP%20Project%20-%20Jon%20Godin%20v01.pdf)


## Conclusions and Recommendations
As mentioned in the Executive Summary, classifying two NFL team-based subreddits proved to be a challenging task, even without removing obvious-classifying tokens (such as Cowboys, Eagles, Dallas, Philadelphia, etc.).

The reason for this is at least two-fold. For one thing, both subreddits consist of fans of NFL football, and therefore use a common vocabulary or terminology that may not differ enough for highly-accurate classification. And for another, I did not scrape a huge sample of posts, so some challenges may be a result of the relatively small sample size used for the analysis.

That said, my best models were able to achieve overall classification accuracy of 66% or better, Sensitivity scores over 70% and ROC AUC scores in the 70s. So the models were decent, though I obviously hoped for better performance. Unfortunately, all of the models showed a relatively large amount of overfitting to the training data.

Models using a Gradient Boosting Classifier were able to achieve the highest overall accuracy, though at the expense of lower Sensitivity. Multinomial Naive Bayes models also performed similarly.

Overall, the models that seemed best able to balance accuracy and Sensitivity were Logistic Regression Classifiers applying L1 (LASSO) regularization.

Differences in word usage were revealed, with strong indicators often being references to various media members who cover one or the other team (such as Justin Melo or Todd Archer). Fans may have been reacting in the form of "did you see what Todd Archer posted?", for example. r/cowboys posts were more likely to contain references to other teams (like Tampa, Dolphins, Steelers, San Francisco). Certain vulgarities were more associated with one or the other subreddit as well.

**Recommendations**

For future research, larger and longer-term data pulls could be attempted to see whether increasing the overall sample size leads to less overfitting and better model performance.

Many posts only consisted of title text without more substantial selftext entries, so another approach would be to screen for posts that contain both title and selftext content.

Lastly, a more rigorous approach to removing stopwords (such as removing ALL current and former player/coach/GM names), and even local sports reporters, might yield more unique topics of interest to each fan base.
