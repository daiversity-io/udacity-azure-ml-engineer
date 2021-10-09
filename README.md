# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This project uses data from direct marketing campaigns of a Portuguese banking institution. The marketing campaigns are based on phone calls. It contains 20 features such as age, job, and marital status. The target column contains only two categories (Yes and No), to determine if the client subscribed to the bank's term deposit. 

The aim of the algorithms built using the Python SDK (w/ Hyperdrive) and AutoML is to accurately predict if a potential client will subscribe to the bank's term deposit. This is to assist in targeting resources in approaching clients that are most likely to subscribe.

The best performing model was found using the AutoML run and was a Voting Ensemble model with an accuracy of 91.642%. 
<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/9732c7935af03290ad7ef7a07779e07abe218d07/images/Screen%20Shot%202021-10-09%20at%208.52.40%20AM.png'>

However, the Logistic classifier trained using Hyperdrive had an accuracy of 91.79% which is very close to the accuracy of that achieved by the Voting Ensemble model.
<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/55c4a0df14e5df074cb7687039645920777b027b/images/Screen%20Shot%202021-10-09%20at%209.31.59%20AM.png'>

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

The best model had parameters of C = 10 and max_iter = 50, and achieved an accuracy of 91.642%.

<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/56fc56999c25611fc22cec442ac9a12277656ed8/images/Screen%20Shot%202021-10-09%20at%208.52.04%20AM.png'>

## AutoML

The steps taken to implement AutoML were as follows:

- Import the banking dataset using Azure TabularDataset Factory
- Data is then cleaned and transformed using the cleaning function in train.py
- AutoML was configured and a run was submitted to find the model with the best performance
- The best model was saved

The best performing model was a Voting Ensemble model with an accuracy of 91.642%. The hyper parameters of the model were as follows:

- max_iter = 100
- multi_class = ovr
- n_jobs = 1
- penalty = 12
- random_state = None
- solver = saga
- tol = 0.0001
- verbose = 0
- warm_start = False

The below snapshots gives the explanation of the best model prediction by highlighting feature importance values and discovering patterns in data at training time. It also shows differnt metrics and their value for model interpretability and explanation

<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/063eaad492d93449196f4e9bccad253d629c4c95/images/Screen%20Shot%202021-10-09%20at%208.58.25%20AM.png'>

<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/063eaad492d93449196f4e9bccad253d629c4c95/images/Screen%20Shot%202021-10-09%20at%209.04.03%20AM.png'>

<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/063eaad492d93449196f4e9bccad253d629c4c95/images/Screen%20Shot%202021-10-09%20at%209.04.39%20AM.png'>

<img src='https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/063eaad492d93449196f4e9bccad253d629c4c95/images/Screen%20Shot%202021-10-09%20at%209.05.47%20AM.png'>

More on voting classifiers can be found in the following links:

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
- https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

## Pipeline Comparison

<p>Both the approaches - Logistics + Hyperdrive and AutoML follow similar data processing steps and the difference lies in their configuration details. In the first approach our ML model is fixed and we use hyperdrive tool to find optimal hyperparametets while in second approach different models are automatic generated with their own optimal hyperparameter values and the best model is selected. In the below image, we see that the hyperdrive approach took overall <b>7m 37s</b> and the best model had an accuracy of <b>~0.9179</b> and the automl approach took overall <b>20m 40s</b> and the best model had an acccuracy of <b>~0.91642</b>.
</p>
<img src = 'https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/7c8266e76c6bc1874c1016117340002914d6b1fc/images/Screen%20Shot%202021-10-09%20at%209.07.14%20AM.png'>

## Future work

<ul>
 <li>To check or measure the fairness of the models</li>
 <li>Leverage additional interactive visualizations to assess which groups of users might be negatively impacted by a model and compare multiple models in terms of their              fairness and performance</li>
</ul>

## Proof of cluster clean up
<img src= 'https://github.com/daiversity-io/udacity-azure-ml-engineer-project-1/blob/7c8266e76c6bc1874c1016117340002914d6b1fc/images/Screen%20Shot%202021-10-09%20at%209.09.04%20AM.png'>
