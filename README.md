# Making Quality Product Recommendations Using TensorFlow
One of the most common applications of machine learning systems is to recommend things to users that they'll be interested in. 
Have you noticed how Spotify and Pandora recommend a certain kind of music, or particular songs or radio stations? 
You may  have observed Netflix recommending movies for you, as well, as in the following screenshot:

![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/netflix.jpeg "Logo Title Text 1")

## Collaborative filtering
Recommendation systems are broadly classified into two categories: content-based filtering and collaborative filtering.

Collaborative filtering algorithms do not need detailed information about the user or the items. 
They build models based on user interactions with items such as song listened, item viewed, link clicked, item purchased or video watched. The information generated from the user-item interactions is classified into two categories: implicit feedback and explicit feedback:
* Explicit feedback information is when the user explicitly assigns a score, such as a rating from 1 to 5 to an item.
* Implicit feedback information is collected with different kinds of interaction between users and items, for example, view, click, purchase interactions in the Retailrocket dataset that we will use in our example.

Further collaborative filtering algorithms can be either user-based or item-based. In userbased algorithms, interactions between users are focused on to identify similar users. Then
the user is recommended items that other similar users have bought or viewed. In itembased algorithms, first, the similar items are identified based on item-user interactions, and
then items similar to the current item are recommended.

# Introducing the Retailrocket dataset
The Retailrocket dataset is available from the Kaggle website, at [kaggle.com](https://www.kaggle.com/retailrocket/ecommerce-dataset).

The Retailrocket dataset comes in three files:
* events.csv: This file contains the visitor-item interaction data.
* item_properties.сsv: This file contains item properties.
* category_tree.csv: This file contains the category tree

The data contains the values collected from an e-commerce website but has beenanonymized to ensure the privacy of the users. The interaction data represents interactions over a period of 4.5 months.

A visitor can engage in three categories of events: view, addtocart, or transaction. The dataset has a total of 2,756,101 interactions that include 2,664,312 view events, 69,332 addtocart events, and 22,457 transaction events. The interactions are from 1,407,580 unique visitors.

Since the data contains the user-item interactions and not the explicit ranking of items by users, it, therefore, falls under the category of implicit feedback information.

# EDA highlights

### Count of Actions
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/Count%20of%20Actions.png "Logo Title Text 1")

### The most viewed itemid
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/top%20viewed%20itemdid.png "Logo Title Text 1")

### How many times an item has displayed
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/number%20of%20times%20item%20apperead.png "Logo Title Text 1")

### Number of total views, number of avg view by top users(quantile 90% and also all users)
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/top%20users-viewed%20item.png "Logo Title Text 1")

### How many times a visitor did an action
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/visitor%20action%20count.png "Logo Title Text 1")

### Top 5 users
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/top5%20users.png "Logo Title Text 1")

### Count per actions over datetimes
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/time%20series.png "Logo Title Text 1")

### Correlation Heatmap
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/Correlation%20Heatmap.png "Logo Title Text 1")

# The matrix factorization model for Retailrocket recommendations
Matrix factorization is a popular algorithm for implementing recommendation systems and
falls in the collaborative filtering algorithms category. In this algorithm, the user-item
interaction is decomposed into two low-dimensional matrices. For example, let's say all the
visitor-item interactions in our dataset are M x N matrix, denoted by A. Matrix factorization
decomposes matrix A into two matrices of M x k and k x N dimensions respectively, such
that the dot product of these two can approximate matrix A. Some of the more popular
algorithms for finding the low-dimensional matrix are based on Singular Value
Decomposition (SVD). In the following example, we'll use the TensorFlow and Keras
libraries to implement matrix factorization.

You can see the layers and output sizes clearly in this plotted visualization:

**mean squared error: 0.9643871370311828**
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/matrix%20factorization.png "Logo Title Text 1")

# The neural network model for Retailrocket recommendations

**mean squared error: 0.05709125054560985**
![alt text](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/Images/neural%20network.png "Logo Title Text 1")

# Summary
We used the Retailrocket dataset to create two models of our recommendation system, one with matrix factorization, and one using a neural network.
We saw that the neural network model gave pretty good accuracy.

[model file](https://github.com/Giats2498/Giats-ecommerce_-recommender_system/blob/master/model).
