import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import os
from django.conf import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scale_features(X):
    """Scale features using StandardScaler"""
    try:
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    except Exception as e:
        logger.error(f"Error scaling features: {str(e)}")
        return X  # Return original if scaling fails

def preprocess_data(file_path, model_type):
    """Preprocess data for different model types"""
    try:
        df = pd.read_csv(file_path)
        
        # Data cleaning and preprocessing
        # Drop rows with missing values
        df = df.dropna()
        
        if model_type == 'linear_regression':
            X = df[['x']].values
            y = df['y'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        elif model_type == 'multiple_regression':
            # Encode categorical variables if present (not in this example)
            X = df[['area', 'bedrooms', 'age']].values
            y = df['price'].values
            
            # Scale the features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        elif model_type == 'polynomial_regression':
            X = df[['x']].values
            y = df['y'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        elif model_type == 'logistic_regression':
            # Encode categorical features if needed
            if 'extracurricular' in df.columns:
                # Binary features are already encoded as 0 and 1
                pass
                
            X = df[['hours_studied', 'previous_score', 'extracurricular', 'sleep_hours']].values
            y = df['passed'].values
            
            # Scale features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        elif model_type == 'knn':
            X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
            
            # Handle categorical target variable
            le = LabelEncoder()
            y = le.fit_transform(df['species'])
            
            # Scale features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def train_linear_regression(X_train, y_train):
    """Train Simple Linear Regression model with cross-validation"""
    try:
        # Create a pipeline with scaling and regularization
        pipeline = Pipeline([
            ('regressor', LinearRegression())
        ])
        
        # Cross-validation for model evaluation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        logger.info(f"Linear Regression CV scores: {cv_scores} with mean: {np.mean(cv_scores)}")
        
        # Train the final model
        pipeline.fit(X_train, y_train)
        return pipeline
    except Exception as e:
        logger.error(f"Error training linear regression: {str(e)}")
        raise

def train_multiple_regression(X_train, y_train):
    """Train Multiple Linear Regression model with regularization"""
    try:
        # Try different regularization methods
        ridge = Ridge(alpha=1.0)
        lasso = Lasso(alpha=0.1)
        linear = LinearRegression()
        
        # Train models
        ridge.fit(X_train, y_train)
        lasso.fit(X_train, y_train)
        linear.fit(X_train, y_train)
        
        # Get cross-validation scores
        ridge_cv = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
        lasso_cv = cross_val_score(lasso, X_train, y_train, cv=5, scoring='r2')
        linear_cv = cross_val_score(linear, X_train, y_train, cv=5, scoring='r2')
        
        # Choose the best model
        best_score = max(np.mean(ridge_cv), np.mean(lasso_cv), np.mean(linear_cv))
        if best_score == np.mean(ridge_cv):
            logger.info(f"Using Ridge Regression with CV score: {np.mean(ridge_cv)}")
            return ridge
        elif best_score == np.mean(lasso_cv):
            logger.info(f"Using Lasso Regression with CV score: {np.mean(lasso_cv)}")
            return lasso
        else:
            logger.info(f"Using Linear Regression with CV score: {np.mean(linear_cv)}")
            return linear
    except Exception as e:
        logger.error(f"Error training multiple regression: {str(e)}")
        raise

def train_polynomial_regression(X_train, y_train, degree=3):
    """Train Polynomial Regression model with cross-validation for degree selection"""
    try:
        best_degree = degree
        best_score = -np.inf
        best_model = None
        best_poly = None
        
        # Try different polynomial degrees if dataset is small enough
        if len(X_train) > 10:  # Only try multiple degrees if we have enough data
            for d in range(1, min(5, degree + 2)):
                poly = PolynomialFeatures(degree=d)
                X_poly = poly.fit_transform(X_train)
                
                model = LinearRegression()
                
                # Cross-validation
                scores = cross_val_score(model, X_poly, y_train, cv=min(5, len(X_train)), scoring='r2')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_degree = d
                    
            logger.info(f"Best polynomial degree: {best_degree} with score: {best_score}")
        
        # Train final model with best degree
        poly = PolynomialFeatures(degree=best_degree)
        X_poly = poly.fit_transform(X_train)
        
        model = LinearRegression()
        model.fit(X_poly, y_train)
        
        return model, poly
    except Exception as e:
        logger.error(f"Error training polynomial regression: {str(e)}")
        # Fall back to linear regression if polynomial fails
        poly = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
        return model, poly

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model with hyperparameter tuning"""
    try:
        # Define a grid of hyperparameters
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
        
        # Create a basic logistic regression model
        lr = LogisticRegression(random_state=42)
        
        # Use grid search for hyperparameter tuning
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, 
                                  cv=min(5, len(X_train)), scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best logistic regression parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error training logistic regression: {str(e)}")
        # Fall back to basic model if grid search fails
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

def train_knn(X_train, y_train, n_neighbors=3):
    """Train K-Nearest Neighbors model with optimal k selection"""
    try:
        # Try different values of k to find the optimal one
        best_k = n_neighbors
        best_score = -np.inf
        
        # Check if we have enough samples for cross-validation
        if len(X_train) >= 10:
            k_range = range(1, min(20, len(X_train) // 2))
            
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, y_train, cv=min(5, len(X_train)), scoring='accuracy')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k
            
            logger.info(f"Best KNN k: {best_k} with score: {best_score}")
        
        # Train model with the best k
        model = KNeighborsClassifier(n_neighbors=best_k)
        model.fit(X_train, y_train)
        
        return model
    except Exception as e:
        logger.error(f"Error training KNN: {str(e)}")
        # Fall back to basic model if optimization fails
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        return model

def evaluate_regression_model(model, X_test, y_test, model_type='linear_regression', poly=None):
    """Evaluate regression models with robust error handling"""
    try:
        if model_type == 'polynomial_regression' and poly is not None:
            X_test_poly = poly.transform(X_test)
            y_pred = model.predict(X_test_poly)
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model evaluation - MSE: {mse}, R²: {r2}")
        
        return {
            'mse': mse,
            'r2': r2,
            'y_pred': y_pred
        }
    except Exception as e:
        logger.error(f"Error evaluating regression model: {str(e)}")
        return {
            'mse': float('inf'),
            'r2': 0,
            'y_pred': None,
            'error': str(e)
        }

def evaluate_classification_model(model, X_test, y_test):
    """Evaluate classification models with robust error handling"""
    try:
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model evaluation - Accuracy: {accuracy}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'y_pred': y_pred
        }
    except Exception as e:
        logger.error(f"Error evaluating classification model: {str(e)}")
        return {
            'accuracy': 0,
            'classification_report': {},
            'y_pred': None,
            'error': str(e)
        }

def get_model_data_path(model_type):
    """Get the data file path for a specific model with error handling"""
    try:
        data_files = {
            'linear_regression': 'linear_regression_data.csv',
            'multiple_regression': 'multiple_regression_data.csv',
            'polynomial_regression': 'polynomial_regression_data.csv',
            'logistic_regression': 'logistic_regression_data.csv',
            'knn': 'knn_data.csv',
            'blood_pressure': 'blood_pressure_data.csv',
        }
        
        file_name = data_files.get(model_type)
        if not file_name:
            logger.error(f"No data file found for model type: {model_type}")
            raise ValueError(f"Invalid model type: {model_type}")
            
        file_path = os.path.join(settings.BASE_DIR, 'ml_app', 'static', 'data', file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        return file_path
    except Exception as e:
        logger.error(f"Error getting model data path: {str(e)}")
        raise

def train_blood_pressure_model(X_train, y_train):
    """Train blood pressure prediction model
    
    This model predicts both systolic and diastolic blood pressure
    based on various health metrics like age, weight, etc.
    """
    try:
        # Try different regression models
        models = {
            'ridge': MultiOutputRegressor(Ridge(alpha=1.0)),
            'lasso': MultiOutputRegressor(Lasso(alpha=0.1)),
            'linear': MultiOutputRegressor(LinearRegression()),
            'random_forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        }
        
        best_score = -np.inf
        best_model_name = None
        
        # Find best model using cross-validation
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), 
                                       scoring='neg_mean_squared_error')
            mean_score = np.mean(cv_scores)
            
            logger.info(f"Blood pressure {name} model CV score: {mean_score}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
        
        # Train final model
        logger.info(f"Using {best_model_name} for blood pressure prediction with score: {best_score}")
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        
        return best_model, best_model_name
    
    except Exception as e:
        logger.error(f"Error training blood pressure model: {str(e)}")
        # Fall back to basic linear regression if other methods fail
        fallback_model = MultiOutputRegressor(LinearRegression())
        fallback_model.fit(X_train, y_train)
        return fallback_model, "linear"

def evaluate_blood_pressure_model(model, X_test, y_test):
    """Evaluate blood pressure prediction model performance"""
    try:
        # Predict blood pressure values
        y_pred = model.predict(X_test)
        
        # Calculate metrics for both systolic and diastolic
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        
        # Calculate mean absolute percentage error
        mape = np.mean(np.abs((y_test - y_pred) / y_test), axis=0) * 100
        
        logger.info(f"Blood pressure model evaluation - MSE: {mse}, MAE: {mae}, R²: {r2}, MAPE: {mape}%")
        
        return {
            'mse': mse.tolist(),  # Convert to list for serialization
            'mae': mae.tolist(),
            'r2': r2.tolist(),
            'mape': mape.tolist(),
            'y_pred': y_pred.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error evaluating blood pressure model: {str(e)}")
        return {
            'mse': [float('inf'), float('inf')],
            'mae': [float('inf'), float('inf')],
            'r2': [0, 0],
            'mape': [float('inf'), float('inf')],
            'y_pred': None,
            'error': str(e)
        } 