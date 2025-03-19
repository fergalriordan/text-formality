# Evaluation System for Formality Detection Models

## Project Overview

The goal of this project is to create an evaluation system for a formality detection model. Formality detection can be framed as a binary classification task, where we train a model to classify whether some sample text either is formal (1) or informal (0). Alternatively, we can view the classification in a continuous manner, where the model measures the formality of text along a continuous spectrum of formality values. In this project, both possibilities are explored, comparing several transformer-based models in this classification task. This README document is structured as follows: 

* Repository Structure
* Dependencies
* Running the Scripts

For a thorough discussion of the work carried out in this project and an analysis of the results, consult the docs/report.pdf document.

## Repository Structure

The repository is structured as follows:

```bash
text-formality/
│── scripts/                        # Scripts for data processing and model evaluation 
│   ├── data_visualisation.ipynb    # Jupyter notebook for preliminary exploration/visualisation of the dataset 
│   ├── data_prep.py                # Script for collecting and preprocessing dataset for subsequent evaluations 
│   ├── evals.ipynb                 # Jupyter notebook for model evaluation and comparison  
│── docs/                           # Documentation and reports  
│   ├── report.pdf                  # Short report describing methodology, results, and challenges  
│   ├── dataset_overview.md         # Detailed description of datasets and sources  
│── README.md
│── requirements.txt
│── .gitignore
```

## Dependencies

To set up the environment and run the scripts, install the required dependencies listed in requirements.txt. The project primarily uses Python and relies on the following libraries:

* Python 3.11.5
* NumPy 
* Pandas
* Matplotlib
* Transformers (Hugging Face) – For working with transformer-based models
* Torch – For deep learning model inference
* Jupyter Notebook – For running exploratory and evaluation notebooks

## Running the Scripts

To reproduce the evaluation process, follow these steps:

### Installing Dependencies

Ensure that all dependencies are installed. If not already installed, use:
```bash
pip install -r requirements.txt
```

### Exploratory Data Visualisation with t-SNE

To gain an understanding of the nature of the dataset through a series of visualisations, run the scripts/data_visualisation.ipynb Jupyter notebook. 

In the Jupyter notebook, the text data is vectorised using the pre-trained GloVe-25 model (trained on a dataset of 2 billion Tweets), and then dimensional reduction is performed on these vectors using t-distributed Stochastic Neighbour Embedding (t-SNE). This facilitates the visualisation of high-dimensional text data through dimensional reduction to 2D or 3D. The goal of this visualisation is to investigate if a clear separation can be seen between the formal and informal instances - if a clear separation can be seen, this suggests that it is possible to train an effective model for classification of this data.

### Model Evaluations

To compare model performance in the classification task, run the evals.ipynb Jupyter notebook.