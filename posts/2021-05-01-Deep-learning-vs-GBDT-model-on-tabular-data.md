# Deep learning vs GBDT model on tabular data
### An experiment of TabNet, MLP and XGBoost performance comparison on the home insurance data set

In the world of data science, deep learning methods are surely state-of-art research. Many new variations of them are invented and implemented every day, especially in the area of NLP (natural language processing) and CV (computer vision), deep learning made so much progress in recent years. This trend can be observed in the Kaggle competitions as well. In those NLP and CV task competitions, recent winning solutions are utilising deep learning models.
However, are deep learning models really better than “traditional” machine learning models like GBDT (gradient boosted decision tree)? We know that, as mentioned above, deep learning models are so much better in NLP and CV, but in the real-life business, we are still having so much tabular data, do we confirm that deep learning models are better performers than GBDT models even on a structured dataset? To answer this question, this post uses the [home insurance](https://www.kaggle.com/ycanario/home-insurance) dataset from Kaggle to compare the performances of each model. I know that we can’t conclude which model is better just by one dataset, but it would be a good starting point to see the comparison. Also, I will use TabNet, which is a relatively new deep learning model for tabular data for comparison.
The notebooks for this experiment can be found in my kaggle notebooks: [main notebook](https://www.kaggle.com/kyosukemorita/deep-learning-vs-gbdt-model-on-tabular-data) and [TabNet with pre-training](https://www.kaggle.com/kyosukemorita/home-insurance-pretrained-tabnet). This post will omit the explanation of each algorithm as there are already plenty of them :)

## Table of contents
1: [Summary of this experiment](#Summary-of-this-experiment)

2: [Model performance](#Model-performance)

3: [Explainability](#Explainability)

4: [Model selection for deployment in the real-life business](#Model-selection-for-deployment-in-the-real-life-business)

5: [Conclusion](#Conclusion)

## Summary of this experiment and code snippet
As mentioned above, this experiment uses the home insurance dataset. This dataset includes home insurance policy data between 2007 and 2012 and there are more than 100 features available regarding home features, owner’s demographics, etc and there are more than 250,000 rows in this data. Using this dataset, this experiment tries to predict whether a home insurance policy is going to lapse. Unfortunately, not all the details of variables in this dataset were given, yet it is good enough to do this experiment.

## Model performance
As mentioned above, here compares the model performance of XGBoost, MLP and TabNet with and without pre-train. The models are evaluated by ROC AUC score and F1 score. F1 scores were calculated at 0.27 as a threshold as I assumed that the distribution of the lapsed insurances is similar to the training distribution. Below is a summary of it.

|                         	| ROC AUC 	| F1 score 	| Time (sec) 	|
|:-----------------------:	|:-------:	|:--------:	|:----------:	|
|         XGBoost         	|  0.7706 	|  0.5591  	|     500    	|
|           MLP           	|  0.7514 	|  0.5458  	|     184    	|
| TabNet without pretrain 	|  0.7579 	|  0.5529  	|    1464    	|
|   TabNet with pretrain  	|  0.7524 	|  0.5484  	|    2370    	|

As we can see, in terms of the accuracy of the model, the XGBoost model is the best one, yet other models are also not far behind it. I have used with and without pretraining for the TabNet model (notebook of without pretraining TabNet can be found [here](https://www.kaggle.com/kyosukemorita/home-insurance-pretrained-tabnet). TabNet with pretraining supposed to be having a better result, but in this dataset, it got a slightly worse result than without pretraining. I am not sure what is exactly the reason but I guess this can be improved by appropriate hyperparameters.

When we look into the distribution of the predictions of each model, we can observe that there are some degrees of similarity between the XGBoost and TabNet model. I guess it might be because TabNet is also using a tree-based-like algorithm. MLP model has a quite different shape compared to other models.

In terms of training time, the MLP model was the fastest one. I have used GPU, so that is the main reason why I got this result. Both TabNet models took quite a long time compared to other models. This makes a lot of differences when it comes to hyperparameter tuning. In this experiment, I didn't do any hyperparameter tuning and used arbitrary parameters. Although MLP's training time is almost 1/3 of the XGBoost model, the number of parameters it needs to optimise is easily more than 10 times of the XGBoost, so if I was doing hyperparameter tuning, it might take longer than the XGBoost model's training with hyperparameter tuning.


## Explainability

Explainability is quite important for some machine learning model business use cases. For example, it is critical to be able to explain why a model is making a particular decision in finance/banking. Imagine that we are deploying a model that can be used for loan approval and a customer wants to know why his application was rejected. Banks can't tell him that we don't know as there are strong regulators in the industry.
Explainability of the model is one of the drawbacks of MLP models. Although we can still evaluate which features contributed to making predictions by using some ways such as using SHAP, it would be more useful if we can check the feature importance list quickly. In this notebook, I will compare only XGBoost and TabNet models' feature importance.

The top 5 important features of the XGBoost model are;

- Marital status - Partner
- Payment method - Non-Direct debit
- Option "Emergencies" included after 1st renewal
- Building coverage - Self-damage
- Option "Replacement of keys" included before 1st renewal

The top 5 important features of the TabNet model without pretraining are;

- Property type 21 (Detail not given)
- "HP1" included before 1st renewal
- Payment method - Pure Direct debit
- Type of membership 6 (Detail not given)
- Insurance cover length in years

Surprisingly, those two models' important features are quite different. The important features from XGBoost are more "understandable and expected" to me - for example, if a customer has a partner, that person should be financially more responsible, thus, the home insurance will less likely to lapse. On the other hand, important features of TabNet are, I would say, less intuitive. The most important feature is "property type 21", where the detail of this feature is not given, so we don't know what is special about this property type. Also the second most important feature, "HP1" included before 1st renewal, where again we don't know what is "HP1". Perhaps, this can be an advantage of TabNet. As it is a deep learning model, it can explore a non-obvious relationship of the features and uses the optimal feature set, especially like this time, where not all the features' details are given.


## Model selection for deployment in the real-life business

When we want to use a machine learning model in real-life business, we need to select the best way to deploy the model and often there are some trade-offs. For example, it is a known fact that when we built a few models like this time and those models' accuracies are quite similar, ensemble them might increase the accuracy. If this ensemble strategy worked perfectly like improved the F1 score by 10%, then it is absolutely necessary to take this strategy, but if this improvement was only 1%, do we still want to take this strategy? Probably not, right? - as running one more model makes the computation more expensive, so usually if the benefit of deploying one more model surpasses the computation cost, we can take this ensemble strategy, otherwise, it is not optimal in terms of business.

Also, regarding the model explainability, whereas the XGBoost model used all 115 features, the TabNet model is only using 16 features (the pre-trained model used only 4 features). This is quite a huge difference and also important to understand those differences. As I mentioned above, in some real-life business use cases, it is critical to know how much contribution those features make. So sometimes although the accuracy is quite high, if the model couldn't explain why it's making that decision, it is difficult to convince people to use it in real life, especially in very sensitive business.

Considering the above 2 points, we would consider that the XGBoost model is superior to other deep learning models in this case. In terms of accuracy, the XGBoost model was slightly better than others (I haven't tried ensembling those predictions from all the models but let's assume, it didn't improve the accuracy much - I might be wrong). And in terms of explainability, as discussed above, the XGBoost model's feature importance list is somewhat we could understand (we can see some logic behind it) and somewhat expected.

## Conclusion

This notebook experimentally compared the model performance of XGBoost, MLP and TabNet on tabular data. Here we are using the home insurance dataset to predict its lapse. As the result of this experiment, we have seen that the XGBoost model has slightly better than other deep learning models in terms of accuracy (F1 score and ROC AUC score), but as this experiment used GPU, the MLP model was the fastest to complete its training. Furthermore, we compared their explainability by seeing the feature importance list of the XGBoost model and TabNet model. The XGBoost model's feature importance list was somewhat more understandable and expected, on the other hand, the TabNet model's one was less intuitive. I think this is caused because of the structure of the algorithm - deep learning models, by nature, explores non-obvious relationships of the features and often it is difficult to understand by a human. From this simple experiment, we confirm that although improvement of deep learning models in recent years is impressive and definitely state-of-the-art, on tabular data, GBDT models are still as good as those deep learning models and sometimes even better than them, especially when we would like to deploy a machine learning model in the real-life business.