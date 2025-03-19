# Project Report: Evaluation System for Formality Detection Models

## Introduction

This document contains a brief discussion of the process behind the implementation of an evaluation system for formality detection models. 

## Justification of the Overall Approach

Formality detection refers to the identification of the level of formality of a text sample. This task can be framed as a regression, where we seek to determine the level of formality along a continuous scale, or as a binary classification task, where we classify instances as belonging to one of two classes: formal or informal. 

For this project, the decision was made to frame formality detection as a binary classification task, as this simplifies the comparison of models and facilitates the use of pre-trained models from the work of Dementieva et al.\cite{dementieva2023detectingtextformalitystudy} which are available on Hugging Face. These models were trained using the GYAFC and X-FORMAL datasets, which are the obvious choice as an available dataset for formality detection. However, access to these datasets must be requested, and despite making a request access to this data was not gained in a timely manner. Instead, an alternative dataset needed to be found. After identifying and analysing such a dataset, it could be used to evaluate selected models from Hugging Face, providing a baseline against which new formality detection models can be compared.

## Methods

### 1. Dataset
The dataset used in this project arises from the work of [CITE], who performed an empirical study of linguistic formality that collected human annotations on sentence-level formality. It includes texts from four distince genres: news, blogs, emails, and online question forums. Each sentence was annotated by 5 human judges using a 7-point Likert scale ranging from -3 (very informal) to +3 (very formal). It is repurposed here for a binary classification task.

The dataset is transformed from a continuous regression task into a binary classification task by applying a threshold to the absolute value of the mean formality scores. This ensures that examples with an ambiguous level of formality are discarded. The filtered dataset is divided into two classes by classifying the remaining instances as either informal or formal based on the negativity/positivity of their scores.

Before applying the pre-trained models to this new binary dataset, some exploratory analysis of the data was performed to investigate the separability of the data based on the two classes. This can be accomplished by first vectorising the data using a pre-trained model, then performing a dimensional reduction of the vectorised so that it can be plotted in two or three dimensions. The pre-trained GloVe-25 model was used for vectorisation, and the dimensionality reduction was performed using t-distributed Stochastic Neighbour Embedding. Note that a comprehensive set of visualisations can be found in the data visualisation Jupyter notebook.

The goal in visualising the data using t-SNE is to investigate if any separation exists between the two classes as a result of the vectorisation performed by the pre-trained GloVe model. If the classes appear to be separable, this indicates that there are underlying differences between the content (semantic or otherwise) in these two classes, and it therefore should be possible to train an effective classifier for this data. Failure of any of the pre-trained models to effectively classify this data is therefore unlikely to be as a result of an issue with the dataset, and instead with the model's ability to generalise to new data. As can be seen in the image below, a clear separation exists between the two classes.

![t-SNE visualization of dataset embeddings for the full dataset (left) and the filtered binary dataset (right). Note that a clear separation between the two classes can be observed. This indicates that it is possible to train an effective classifier for this dataset, and that it is an appropriate dataset for the evaluation of pre-trained classifiers.](../images/Untitled%20design.PNG)

### 2. Metrics
binary classification metrics (accuracy, precision, recall, f1)
consideration of regression, comparison of logits with formality scores, spearman correlation etc

### 3. Models
models for binary classification
transformer-based approaches form the sota in this domain
models available on hugging face from paper

## Challenges Faced

(methods)
other dataset not readily available (requested from Yahoo but not approved yet) 
computational resources (GPU, decided to use Colab and only use the test split of the dataset, but could have used the full thing with more cpu power)

## Results and Conclusions

model 3 is the best as expected
these provide a solid baseline for comparison with new models

A solid evaluation system has been implemented
The dataset is clearly suitable for the task
Users simply need to import the new model in the evals file
Would be interesting to explore as a regression task or to incorporate the logits into the evaluation in some way
could have fine-tuned the models on this data, but again time and resource constraints
noteworthy that they perform so well on unseen data (results are similar to evals in original paper)