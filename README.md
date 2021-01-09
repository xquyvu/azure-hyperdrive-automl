# Optimising an ML Pipeline in Azure

## Overview

In this project, I built and optimized an Azure ML pipeline using the Python SDK and a provided Scikit-learn model (Logistic regression).
This model was then compared to an Azure AutoML run.

**Caveat**: The aim of this project was to practice developing ML pipelines using Azure HyperDrive and Azure AutoML, not to achieve the best accuracy nor applying proper data science methodology to problem solving. Therefore, data ETL, EDA, evaluation, etc. which are major components of a data science projects were skipped or not invested in.

## Summary

The dataset resulted from a phone-based marketing campaign of a banking institution in Portugal. In this marketing campaign, potential customers were contacted by phone and introduced to the bank's product (bank term deposit) and they can decide to subscribe to the product or not.

In this project, we try to predict how likely is a customer to subscribe to the product given

- Their demographic profile (age, job, loan, ...)
- The context of the phone call (time, duration, ...),
- Socio-economic context (employment variation rate, CPI, ...)
- Other variables

The best predictor was a VotingEnsemble, built using Azure AutoML which achieved a ~92% accuracy on a 4-fold cross validation.

## Scikit-learn Pipeline

The pipeline's components are as follows:

- Data loading: Azure `TabularDatasetFactory` was used to load the data from an online blob storage
- Data preprocessing and cleaning: `Pandas` was utilised to drop NAs and one hot encode categorical variables.
- Classification algorithm: Logistic regression. This model is simple and light weight enough to be used as baseline for this project.
- Hyper parameter tuning: Azure `Hyperdrive`, which composes of 2 components:

  - Parameter sampler: `RandomParameterSampling` class. This is used to select values for 2 important hyperparameters: `C` which is the inverse of regularisation strength, and `max_iter` for maximum of iterations.
  - Optimisation policy: Bandit Policy with a slack factor of 0.1 and evaluation interval of 1. Since this policy terminates runs where the defined metric (Accuracy) is not within the specified slack factor compared to the best performing run, thus save time and computation resources

## AutoML Pipeline

In this pipeline, no preprocessing step was necessary. All it took was to point the pipeline to a data source and specify some configurations as follows:

```python
AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    n_cross_validations=4,
    max_concurrent_iterations=4,
    compute_target=target,
    enable_early_stopping=True,
    iterations=15,
    blocked_models=['SVM']
)
```

In this configuration, I explicitely block Support Vector Machine because it is extremely slow (15x comparing to other models).

After iterating through 15 models, the AutoML pipeline selected the VotingEnsemble model as the best performing. VotingEnsemble models can be considered as a meta-model or a model-of models, since it combines the predictions from multiple other models.

## Pipeline comparison

In this project, accuracy was selected as the primary metric by Udacity.

Based on this metric, AutoML performed slightly better than the SKLearn pipeline (0.917 vs 0.913). This is due to the dataset being highly imbalance with the ratio being approximately 1:10. Therefore, a model can simply achieve ~90% accuracy by only output the majority class. With that being said, accuracy was an inappropriate metrics to evaluate the pipelines' performance.

Nevertheless, AutoML utilises a large range of complex models and sophisticated hyperparameters tuning techniques. Therefore, given the same set of data, it is expected to perform better than a simple logistic regression model selected by the SKLearn pipeline.

## Future work

- Replace `SKLearn` estimator with `ScriptRunConfig`. Although they have similar functionalities, `SKLearn` API is now deprecated and should be replaced with `ScriptRunConfig` which is more generic.
- Re-evaluate the 2 pipelines using AUROC and AUPRC
