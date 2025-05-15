# Practical Applications in Machine Learning 

# Structure-Based Classification of Antigen Presentation and T Cell Recognition

T cell activation is triggered by the binding of T cell receptors (TCR) to peptide antigens presented by major histocompatibility complex (MHC) proteins. Due to the critical role T cells play in recognizing and responding to infection, significant efforts have been made to understand and predict MHC presentation and TCR specificity to advance immunotherapy. The goals of this project were to utilize structure-based metrics of pMHC-TCR complexes generated using AlphaFold3, a deep learning model, to predict MHC presentation of antigens and TCR recognition of pMHC complexes. At this time, we have not completed our plans for classifying TCR specificity but did complete our evaluation for MCH presentation. We trained models using two supervised machine learning (ML) approaches, naive Bayes classification and support vector machines (SVM), and evaluated performance using accuracy, precision, recall, and F1 score. Our naive Bayes model achieved an accuracy of 0.86 and F1 score of 0.92 on our test dataset while our SVM model with hinge loss using the soft margin method showed slightly improved performance with an accuracy and F1 score of 0.89 and 0.94. Regarding impact, our project demonstrates the utility of easily interpretable structural metrics, as opposed to sequence-based measures, for predicting MHC presentation, and suggests these metrics may be useful to predict TCR specificity.

Please view our notebook (experiments.ipynb) for full experimental details. Our dataset is included in the 'datasets' folder, and the Streamlit pages, excluding the Home page, is in the 'pages' folder. helper_functions.py includes functions for loading the dataframe between pages and reducing dimensionality with PCA.

# Streamlit Page Descriptions

* <b>Home<b>: Information on project and pMHC-TCR binding
* <b>Visualize<b>: Generate plots for data exploration and preprocess data
* <b>Classify<b>: Train and evaluate model using best hyperparameters chosen from cross-validation
* <b>Info<b>: References

To test a different dataset, change
```
dataset_filename = './datasets/MelissaPathil_pMHC_TCR_dataset.csv'
```
Upload a CSV with the same features included in the original dataset.

# Running Streamlit Application

Prerequisites: Install [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/).
To install all the required packages. Use this command in your terminal:
```
conda env create -f environment.yml
```
And to activate the environment:
```
conda activate info5368
```

```
To run the Streamlit app, use the following command:
```
streamlit run 1_Home.py
```
Code based on class and homework notes from Cornell Tech PAML Spring 2025.
