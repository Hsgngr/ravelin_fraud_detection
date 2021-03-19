# Ravelin Data Science Test
## Introduction
Hi, Thanks for reading the report. This report outlines my approach to Ravelin's Fraud Detection data science test. The goal of the project was to correctly classify customer's label as
"fraudulent" or not. We are given 168 customers with their orders, payment methods and transactions. As we are deciding whether a customer is a fraud or not, the project falls under the category of binary classification problem.

The main challenges of the project are three-fold:

* The given data (customer.json) is deeply nested and there is no easy nor direct way to flatten the data to dataframe. 
* The task highly depends on feature engineering and how to combine informations of orders with customers.
* The dataset size is small which creates a risk of overfitting.

## Approach

### Exploratory Data Analysis
In this section I highlighted the things that I find interesting and useful about the data. I also write some reccommendations for preprocessing section.

![image](https://user-images.githubusercontent.com/32769732/111234838-ab50cb00-85ac-11eb-89ec-8323300fd406.png)

There are 107 normal and 61 fraudulent customers. The data has imbalance.

2) DataFrame for Customers:
3) 
![image](https://user-images.githubusercontent.com/32769732/111549554-9d7c8080-8739-11eb-9b3a-cc06177a833f.png)

*customerEmail* and *customerIPAdress* need to be preprocessed as *email_domain* and *IpAddress_Type*. There are 3 duplicants with the same IPaddress.

3) DataFrame for Orders: 

![image](https://user-images.githubusercontent.com/32769732/111549650-c69d1100-8739-11eb-9202-65a94a8e715c.png)

*OrderId* is important to match the orders with transactions. *orderState* should be engineered as categorical feature. *orderAmount* also may give some helpful information about the order and the customer.

Although there are only 168 customers there are 478 orders. Here are stacked histograms to show number of orders per each customer :

Counting Every Order         
:-------------------------:|
![image](https://user-images.githubusercontent.com/32769732/111232089-a0476c00-85a7-11eb-87df-18ff00a58241.png)

Fraudulent customers spread homogeneously through data.

4) DataFrame for Transactions:

![image](https://user-images.githubusercontent.com/32769732/111549698-e16f8580-8739-11eb-9fdc-185250ea74e0.png)


5) DataFrame for PaymentMethods:

![image](https://user-images.githubusercontent.com/32769732/111549744-f64c1900-8739-11eb-8ea8-b3e475257c04.png)

* I have found that for all orders *transactionAmounts* are equal between transactions.
* I have found that all transactions which has same *orderShippingAdresses* belongs to the same *orderId*.

### Preprocessing
After expanding all of the datasets it is time to merge them by using identical columns or creating custom features.
I merged the transactions which has the same orderId into one row. For that I created a custom function to create following features:

*  orderID
*  transaction_count
*  orderAmount
*  fullfilled_order_count
*  pending_order_count
*  failed_order_count
*  orderShippingAddress
*  paymentMethod_count
*  transactionFailed_count
*  paymentMethodProvider_count
*  paymentMethodType_apple
*  paymentMethodType_btc
*  paymentMethodType_card
*  paymentMethodType_paypal

I created a dataframe which has 478 rows × 14 columns.

![image](https://user-images.githubusercontent.com/32769732/111549869-36ab9700-873a-11eb-8abe-7ffd8a4980cd.png)

Then again I created a custom merging function to group orders and add to each customer. Here are the features that I created from them:

* customerEmail
* customerPhone
* customerDevice
* customerIPAddress
* order_counts
* transaction_counts
* orderAmount_sum
* orderAmount_mean
* fulfilled_order_counts
* pending_order_counts
* failed_order_counts
* shipping_adresses_count
* shipping_adress_same
* paymentMethod_counts
* paymentMethodProvider_counts
* paymentMethodType_apples
* paymentMethodType_btcs
* ,paymentMethodType_cards
* paymentMethodType_paypals

These features were mostly sum of the features that I extracted before. Thats why most of them as 's on their name. For example it was *fullfilled_order_count*
and I sum them and created *fullfilled_order_counts*

Before starting the training I changed customerEmail to customerDomain, customerPhone to customerPhoneType, customerIPAddress to customerIPAddressType and deleted the customerDevice

![image](https://user-images.githubusercontent.com/32769732/111549441-67d79780-8739-11eb-8476-42fd42647030.png)

Changed the *customerEmail* to **other** if it is not one of the main 3 provider (*gmail* , *yahoo*, or *hotmail*)

Dropped *CustomerPhone* and *customerDevice* columns

Changed the *CustomerIPAddress* to two different categories as **digits** and **digits_and_letters**

![image](https://user-images.githubusercontent.com/32769732/111549376-4d9db980-8739-11eb-90ee-8b8db7ea829b.png)

This shows how many orders have one transaction. Added as *order_transaction_same* column

![image](https://user-images.githubusercontent.com/32769732/111550114-ae79c180-873a-11eb-89ea-d27c811e9d62.png)

In the end of our preprocessing section we have a dataset of 168 rows x 28 columns including *fraudulent* column.

![image](https://user-images.githubusercontent.com/32769732/111551039-8ee39880-873c-11eb-88ce-451f7c4bbbb0.png)


### Main Approach
I split the train and test set using sklearn's *train_test_split* method by stratify the dataset equally. The test size is %20 of all data.
I ran one of my custom feature importance scripts. Internally it uses an lightgbm to calculate importance of each feature and then I plotted:

![image](https://user-images.githubusercontent.com/32769732/111553837-63fc4300-8742-11eb-817f-2ccb320c010c.png)

Very interestingly the least column I suspect was orderAmounts but they turned to be the most important feature. Here is the cumulative importance of my features:

![image](https://user-images.githubusercontent.com/32769732/111717448-477d0b00-8815-11eb-8e0d-3d4f4aeea229.png)

15 features required for 0.95 of cumulative importance.

For training I tried Support Vector Classifier and RandomForestClassifier, I didnt want to work with Deep Learning since we don't have enough data to train a better neural network than these relatively simpler algorithms.

## Results and Discussion

The accuracy got reached to %79 with SVC() and %73 with Random Forest Classifier. But in systems such as fraud detection the accuracy metric is not enough to measure model's performance.
Here is the confusion matrix of the model:

True Positive: 22          | False Negative: 0
:-------------------------:|:-------------------------:
False Positive: 7          | True Negative: 5

The AUC score is 0.70

![image](https://user-images.githubusercontent.com/32769732/111717064-8eb6cc00-8814-11eb-9f63-a72e12bbac66.png)

![image](https://user-images.githubusercontent.com/32769732/111722508-36d19280-881f-11eb-8d17-1e7192f81b7d.png)




## Conclusion and Future Work
I followed the rules of the test and get things done as soon as possible to create an end-to-end solution. All of the sections: EDA, Preprocessing and Training happened in one iteration so I would definetly go back and play with the feature engineering a bit more after I learned orderAmounts are really important. Then in preprocessing part I can come up with ideas about the columns that I had to drop for example *customerPhone* and *customerDevice*. In the training part, I would try to balance the data by using upsampling techniques so I can have more data for the training. I would use gridSearch to tune hyperparameters of my learning algorithm.
