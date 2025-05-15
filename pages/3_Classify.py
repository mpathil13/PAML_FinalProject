import pandas as pd 
import numpy as np   
import streamlit as st                  
import random                 
from sklearn.model_selection import train_test_split
from helper_functions import fetch_dataset, reduce_feature_dimensionality
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

# Set random seed=10 to produce consistent results
random.seed(10)

#############################################

st.markdown('# Classify')

#############################################

def split_dataset(df, number, target, random_state=42):
    """
    Splits the dataset into training and test sets based on a given ratio
    Inputs:
        - df: DataFrame containing dataset
        - number: Percentage of data to be used for testing
        - target: Name of the target column ('rating')
        - random_state: Seed for reproducibility
    Outputs:
        - X_train: Training feature set (encoded)
        - X_test: Test feature set (encoded)
        - y_train: Training target labels
        - y_test: Test target labels
    """
    try:
        X, y = df.loc[:, ~df.columns.isin(
            [target])], df.loc[:, df.columns.isin([target])]

        # Split the train and test sets into X_train, X_val, y_train, y_val using X, y, number/100, and random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=number/100, random_state=random_state)

        # Compute and print dataset percentages
        train_percentage = (len(X_train) /
                            (len(X_train)+len(X_test)))*100
        test_percentage = (len(X_test) /
                           (len(X_train)+len(X_test)))*100
        st.markdown('The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} \
                    observations ({3:.2f}%).'.format(len(X_train), train_percentage, len(X_test),  test_percentage))
    except:
        print('Exception thrown; testing test size to 0')
    st.markdown(":penguin: Performed principal components analysis (PCA) to reduce dimensionality.")
    
    # Reduce feature dimensionality with PCA
    X_train, X_test = reduce_feature_dimensionality(X_train, X_test)

    return X_train, X_test, y_train, y_test

def k_fold_cross_validation(X_train_data, k=5):
    """
    Performs k-fold cross-validation.
    Inputs:
        - X_train_data: Feature training data
        - k: Number of folds
    Outputs:
        list: A list of tuples, where each tuple contains:
            - train_indices: Indices for the training
            - test_indices: Indices for the test
    """
    num_samples = len(X_train_data)
    fold_size = num_samples // k
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  

    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, val_indices))
    return folds

# Naive Bayes class
class NaiveBayes(object):
    def __init__(self, classes, alpha=1):
        self.model_name = 'Naive Bayes'
        self.classes = classes
        if not isinstance(self.classes, np.ndarray):
            self.classes = np.array(self.classes)
        self.num_classes = len(self.classes)
        mapping = {i: k for i, k in enumerate(self.classes)}
        self.idx_to_class = np.vectorize(mapping.get)
        self.likelihood_history=[]
        self.alpha = alpha
        self.W=[]
        self.W_prior=[]
    
    def predict_logprob(self, X):
        """
        Computes the log probability of each class given input features
        Inputs:
            - X: Input features
        Outputs:
            - y_pred: log probability of positive product review
        """
        # ensure features are positive for NB
        X = X - X.min()
        y_pred=None
        try:
            # Compute log probability using dot product of features 
            #   with log of class conditional probabilities
            y_pred = np.dot(X, np.log(self.W.T))
            # Add log prior probabilities of classes
            y_pred += np.log(self.W_prior)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    def predict_probability(self, X):
        """
        Produces probabilistic estimate for P(y_i = +1 | x_i)
        Inputs:
            - X: Input features
        Outputs:
            - y_pred: probability of positive product review (range: 0 to 1)
        """
        # ensure features are positive for NB
        X = X - X.min()
        y_pred=None
        try:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            # Identify index for positive class
            pos_class_idx = np.where(self.classes == 1)[0][0]
            # Get log probabilities
            y_pred = self.predict_logprob(X)
            # Convert log probabilities to probabilities using the softmax function
            probs = np.exp(np.array(y_pred))
            probs = np.exp(probs) / np.sum(np.exp(probs),axis=1)[:, None]
            # Extract probability of positive class
            y_pred = probs[:, pos_class_idx]
        except ValueError as err:
            st.write({str(err)})
        return y_pred

    def predict(self, X):
        """
        Predicts the class label for each input sample
        Inputs: 
            - X: Input features
        Outputs:
            - y_pred: List of predicted class labels
        """
        y_pred=None
        
        # Ensure features are positive for NB
        X = X - X.min()
        
        try:
            # Compute log probabilities
            y_pred = self.predict_logprob(X)
            # Select class with highest probability
            y_pred = np.argmax(y_pred, axis=1)
            # Convert indices to class labels
            y_pred = self.idx_to_class(y_pred)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    def fit(self, X, Y):
        """
        Initialize self.num_examples, self.num_features, weights self.W, prior probabilities self.W_prior 
            Fits the Naive Bayes classifier using a closed-form solution
        Inputs: 
            - X: Input features
            - Y: list of actual product sentiment classes 
        Outputs:
            - self: The trained Naive Bayes model
            - self.W: fitted model weights
            - self.likelihood_history: history of log likelihood
        """
        try:
            # Number of examples, Number of features
            self.num_examples, self.num_features = X.shape
            # Initialization: weights, prior
            self.W = np.zeros((self.num_classes, self.num_features))
            self.W_prior = np.zeros(self.num_classes)
            # Ensure features are positive
            X = X - X.min()
            # closed-form solution for model parameters
            # Compute class-conditional probabilities and class priors
            for ind, class_k in enumerate(self.classes):
                # Select samples belonging to class_k
                X_class_k = X[Y == class_k]
                # Compute likelihood
                self.W[ind] = (np.sum(X_class_k, axis=0) + self.alpha)
                self.W[ind] /= (np.sum(X_class_k) + (self.alpha * X_class_k.shape[-1]))
                # Compute prior
                self.W_prior[ind] = X_class_k.shape[0] / self.num_examples
            # Compute and store log likelihood history
            log_likelihood = np.log(self.predict_probability(X)).mean()
            self.likelihood_history.append(log_likelihood)
        except ValueError as err:
            st.write({str(err)})
        return self

# Support Vector Machine (SVM) Class
class SVM(object):
    def __init__(self, learning_rate=0.001, num_iterations=500, lambda_param=0.01):
        self.model_name = 'Support Vector Machine'
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.likelihood_history = []
        self.W=[]
        self.b=0
    
    def predict_score(self, X):
        """
        Produces raw decision values before thresholding
        Inputs:
            - X: Input features
        Outputs:
            - scores: Raw SVM decision values
        """
        scores=None
        try:
            # Take dot product of feature_matrix and coefficients  
            scores = np.dot(X, self.W) + self.b
        except ValueError as err:
            st.write({str(err)})
        return scores
    
    def compute_hinge_loss(self, X, Y):
        """
        Compute the hinge loss for SVM using X, Y, and self.W
        Inputs:
            - X: Input features
            - Y: Ground truth labels
        Outputs:
            - loss: Computed hinge loss
        """
        loss=None
        try:
            # Compute margin values
            margins = 1 - Y * self.predict_score(X)
            # Apply hinge loss condition
            margins = np.maximum(0, margins)
            # Regularized loss
            loss = np.mean(margins) + (self.lambda_param / 2) * np.sum(self.W ** 2)
        except ValueError as err:
            st.write({str(err)})
        return loss

    def update_weights(self):
        """
        Compute SVM derivative using gradient descent and update weights
        Inputs:
            - None
        Outputs:
            - self: The trained SVM model
            - self.W: Weight vector updated based on gradient descent
            - self.b: Bias term updated based on gradient descent
            - self.likelihood_history: history of log likelihood
        """
        try:
            # Compute decision scores
            scores = self.predict_score(self.X)
            # Identify misclassified points
            indicator = (self.Y * scores) < 1
            # Compute gradients for weights and bias
            dW =( -np.dot(self.X.T, (self.Y * indicator)) + (2 * self.lambda_param * self.W) ) / self.num_examples
            db = -np.sum(self.Y * indicator) / self.num_examples
            # Update weights and bias parameters using gradient descent
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            # Compute and store loss history
            loss = self.compute_hinge_loss(self.X, self.Y)
            self.likelihood_history.append(-loss)
        except ValueError as err:
            st.write({str(err)})
        return self
    
    def predict(self, X):
        """
        Predicts class labels using the trained SVM model
        Inputs:
            - X: Input features
        Outputs:
            - y_pred: List of predicted classes (-1 or +1)
        """
        y_pred=None
        try:
            # Compute raw decision scores
            scores = self.predict_score(X)
            # Convert scores to class labels
            y_pred = np.where(scores >= 0, +1, 0)
        except ValueError as err:
            st.write({str(err)})
        return y_pred
    
    def fit(self, X, Y):
        """
        Train SVM using gradient descent.
        Inputs:
            - X: Input features
            - Y: True class labels (-1 or +1)
        Outputs:
            - self: Trained SVM model
        """
        try:
            # Number of examples, Number of features
            self.num_examples, self.num_features = X.shape
            # Initialization: weights, features, target, bias, likelihood_history
            self.W = np.zeros(self.num_features)
            self.b = 0
            self.X = X
            self.Y = Y
            self.likelihood_history = []
            # Perform gradient descent updates for specified number of iterations
            for _ in range(self.num_iterations):
                self.update_weights()
        except ValueError as err:
            st.write({str(err)})
        return self
    
def compute_evaluation_metrics(X, y_true, model): 
    """ Compute classificaition accuracy, precision, recall, F1 score
        Input
        - prediction_labels (numpy): predicted product sentiment
        - true_labels (numpy): true product sentiment
        Output
        - accuracy (float): accuracy percentage (0-100%)
        - precision (float): precision score = TP/TP+FP
        - recall (float): recall score = TP/TP+FN
        - F1 score (float): F1 score = 2 * precision * recall / (precision + recall)
    """
    metric_dict = {}
    prediction_labels = model.predict(X)
    true_labels=  np.array(np.ravel(y_true), dtype=np.int32)  

    cmatrix = confusion_matrix(true_labels, prediction_labels)
    true_neg, false_pos, false_neg, true_pos = cmatrix.ravel()

    # calculate accuracy
    num_correct = np.sum(prediction_labels == true_labels)
    metric_dict['accuracy'] = num_correct / len(true_labels) 
    
    # calculate precision
    precision = true_pos / (true_pos + false_pos)
    metric_dict['precision'] = precision
    
    # calculate recall
    recall = true_pos / (true_pos + false_neg)
    metric_dict['recall'] = recall
    
    # calculate F1 score
    metric_dict['F1_score'] = 2 * (precision * recall) / (precision + recall)
    return metric_dict

###################### FETCH DATASET #######################
df = None
df = fetch_dataset()

if df is not None:

    # Display dataframe as table
    st.dataframe(df)

    if 'presenter_category_target' not in df.columns:
        st.markdown("Uh-oh! We're missing our target. Preprocess the data in `Visualize` to continue.")
    else:
        feature_predict_select = 'presenter_category_target'
        st.session_state['target'] = feature_predict_select

        st.markdown('#### :whale: TRAIN/TEST SPLIT')

        # Compute the percentage of test and training data
        if (feature_predict_select in df.columns):
            X_train, X_test, y_train, y_test = split_dataset(
                df, 20, feature_predict_select)
            
            # Save train and test split to st.session_state
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test

        classification_methods_options = ['Naive Bayes', 
                                        'Support Vector Machine']
        
        st.markdown('#### :whale: MODEL TRAINING')

        # Collect ML Models of interests
        classification_model_select = st.multiselect(
            label='Select regression model for prediction',
            options=classification_methods_options,
        )
        st.write('You selected the following models: {}'.format(
            classification_model_select))

        # Training Naive Bayes
        if (classification_methods_options[0] in classification_model_select):
            st.markdown('#### :octopus: **Naive Bayes**')
            if st.button('Train Naive Bayes'):
                try:
                    nb_model = NaiveBayes(np.unique(np.ravel(y_train)))
                    nb_model.fit(X_train, np.ravel(y_train))
                    st.session_state[classification_methods_options[0]] = nb_model
                except ValueError as err:
                    st.write({str(err)})
            
            if classification_methods_options[0] not in st.session_state:
                st.write('Naive Bayes Model is untrained')
            else:
                st.write('Naive Bayes Model trained')

        # Training SVM
        if (classification_methods_options[1] in classification_model_select):
            st.markdown('#### :octopus: **Support Vector Machine**')

            svm_params = {
                'num_iterations': 10,
                'learning_rate': 0.1,
                'lambda_param': 0.5
            }

            if st.button('Train SVM Model'):
                try:
                    svm_model = SVM(num_iterations=svm_params['num_iterations'], 
                                    learning_rate=svm_params['learning_rate'],
                                    lambda_param=svm_params['lambda_param'])
                    svm_model.fit(X_train, np.ravel(y_train))
                    st.session_state[classification_methods_options[1]] = svm_model
                except ValueError as err:
                    st.write({str(err)})

            if 'Support Vector Machine' not in st.session_state:
                st.write('SVM Model is untrained')
            else:
                st.markdown(':penguin: SVM Model trained with the following hyperparameters: \
                \n - learning rate: 0.1 \
                \n - number of iterations: 10 \
                \n - lambda: 0.5 ')
                
        # Store models in dict
        trained_models={}
        for model_name in classification_methods_options:
            if(model_name in st.session_state):
                trained_models[model_name] = st.session_state[model_name]

        if 'Support Vector Machine' in st.session_state:
            # Inspect model likelihood
            st.markdown(':octopus: View SVM Model Likelihood')

            # Display SVM likelihood
            inspect_model_likelihood = 'Support Vector Machine'
            if inspect_model_likelihood in trained_models:
                try:
                    fig = make_subplots(rows=1, cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
                    cost_history=trained_models[inspect_model_likelihood].likelihood_history

                    x_range = (0, len(cost_history))
                    cost_history_tmp = cost_history[x_range[0]:x_range[1]]
                    
                    fig.add_trace(go.Line(x=np.arange(x_range[0],x_range[1],1),
                                y=cost_history_tmp, mode='lines+markers', name=inspect_model_likelihood), row=1, col=1)

                    fig.update_xaxes(title_text="Training Iterations")
                    fig.update_yaxes(title_text='Log Likelihood', row=1, col=1)
                    fig.update_layout(title=inspect_model_likelihood)
                    st.plotly_chart(fig)
                except Exception as e:
                    print(e)

        st.markdown('#### :whale: MODEL EVALUATION')

        metric_options = ['accuracy', 'precision', 'recall', 'F1_score']
        trained_models = [
            model for model in classification_methods_options if model in st.session_state]
        st.session_state['trained_models'] = trained_models

        # Select a trained classification model for evaluation
        model_select = st.multiselect(
            label='Select trained classification models for evaluation',
            options=trained_models
        )
        if (model_select):
            st.write(
                'You selected the following models for evaluation: {}'.format(model_select))

            eval_button = st.button('Evaluate your selected classification models')

            if eval_button:
                st.session_state['eval_button_clicked'] = eval_button

            if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
                models = [st.session_state[model]
                            for model in model_select]

                train_result_dict = {}
                val_result_dict = {}

                for idx, model in enumerate(models):
                    train_result_dict[model_select[idx]] = compute_evaluation_metrics(
                        X_train, y_train, model)
                    val_result_dict[model_select[idx]] = compute_evaluation_metrics(
                        X_test, y_test, model)

                st.markdown('#### Performance on the training dataset')
                st.dataframe(train_result_dict)

                st.markdown('#### Performance on the test dataset')
                st.dataframe(val_result_dict)
