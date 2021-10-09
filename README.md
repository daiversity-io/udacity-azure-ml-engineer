# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This project uses data from direct marketing campaigns of a Portuguese banking institution. The marketing campaigns are based on phone calls. It contains 20 features such as age, job, and marital status. The target column contains only two categories (Yes and No), to determine if the client subscribed to the bank's term deposit. 

The aim of the algorithms built using the Python SDK (w/ Hyperdrive) and AutoML is to accurately predict if a potential client will subscribe to the bank's term deposit. This is to assist in targeting resources in approaching clients that are most likely to subscribe.

The best performing model was found using the AutoML run and was a Voting Ensemble model with an accuracy of 91.78%. However, the Logistic classifier trained using Hyperdrive had an accuracy of 91.44% which is very close to the accuracy of that achieved by the Voting Ensemble model.

## Scikit-learn and Hyperdrive Pipeline

### Scikit-learn

A Logistic Regression model was first created and trained using Scikit-learn in the train.py. The steps taken in the python script were as follows:

- Import the banking dataset using Azure TabularDataset Factory

- Data is then cleaned and transformed using a cleaning function

- Processed data is then split into a training and testing set

- Scikit-learn was used to train an initial Logistic Regression model while specifying the value of two hyper parameters, C and max_iter. C represents the inverse of the regularization strength, while max_iter represents the maximum number of iterations taken for the model to converge. These two parameters were initially passed in the python script so they can be optimised later using Hyperdrive.

- The trained model is then saved

The model had parameters of C = 0.1 and max_iter = 100, and achieved an accuracy of 91.43%

### Hyper Drive

The initial model trained is then optimised using Hyperdrive. Hyperdrive is a method of implementing automatic hyperparameter tuning. Hyperparameter tuning is typically computationally expensive and manual, therefore, by using Hyperdrive we can automate this process and run experiments in parallel to efficiently optimize hyperparameters.

The steps taken to implement Hyperdrive were as follows:

- Configuration of the Azure cloud resources

- Configuring the Hyperdrive

- Running the Hyperdrive

- Retrieving the model with the parameters that gave the best model

Elaborating more on the second step in configuring the Hyperdrive, there are two extremely beneficial parameters that are included in the configuration; RandomParameterSampling and BanditPolicy.

**RandomParameterSampling** is a parameter sampler that randomly selects hyperparameter values from a wide range specified by the user to train the model. This is much better than a grid sweep as it is not as computationally expensive and time-consuming and can choose parameters that achieve high accuracy. Random sampler also supports early termination of low-performance runs, thus saving on computational resources. The parameters passed to the random sampler were:

- C: 0.01,0.1,10,100

- max_iter: 50,100,150,200

**BanditPolicy** is an early termination policy that terminates runs early if they are not achieving the same performance as the best model. This also adds to improving computational efficiency and saving time as it automatically terminates models with a poor performance.

The best model had parameters of C = 10 and max_iter = 50, and achieved an accuracy of 91.44%.

## AutoML

The steps taken to implement AutoML were as follows:

- Import the banking dataset using Azure TabularDataset Factory
- Data is then cleaned and transformed using the cleaning function in train.py
- AutoML was configured and a run was submitted to find the model with the best performance
- The best model was saved

The best performing model was a Voting Ensemble model with an accuracy of 91.78%. The hyper parameters of the model were as follows:

- max_iter = 100
- multi_class = ovr
- n_jobs = 1
- penalty = 12
- random_state = None
- solver = saga
- tol = 0.0001
- verbose = 0
- warm_start = False

More on voting classifiers can be found in the following links:

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
- https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

## Pipeline Comparison
When comparing both pipelines together, AutoML seems to have the advantage due to:

- Less steps taken to find the best model (Simpler architecture)
- Achieved better accuracy

I think the main advantage of automl compared to hyperdrive is the ability of automl to test different algorithms easily. We might think that the model chosen was the best for this problem and try to optimize the hyperparameters using hyperdrive. However, there might be a model we haven't tested that might perform better than the model we chose, which is what happened in this project. 

## Future work

The main area of improvement is to take the voting ensemble algorithm from the AutoML run and tune the hyper parameters using Hyperdrive. AutoML uses Bayesian Optimization to choose the best hyper parameters. It would be beneficial to use hyperdrive and try different parameter sampling methods including random and grid parameter sampling. These methods might detect a different set of hyper parameters that give a better accuracy than those chosen by AutoML.

We know that the bank is particularly interested in accurately identifying clients that are more willing to subscribe. Additionally, identifying someone that is not willing to subscribe to the bank’s deposit as someone who does would be detrimental to the bank as it will be a waste of resources. Therefore, a model's ability to precisely predict those who are willing to donate is more important than the model's ability to recall those individuals. Thus, we can use F-beta score as a metric that considers both precision and recall: 

![f-beta score](https://github.com/adhamalhossary/optimizing-a-machine-learning-pipeline-in-azure/blob/main/f-beta%20score.svg)

Using a beta of 0.5 will place more emphasis on precision. We would then use the F-beta score instead of accuracy to test both the hyperdrive to see which hyper parameters give the best f-beta score, and which model from AutoML would as well.



# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
