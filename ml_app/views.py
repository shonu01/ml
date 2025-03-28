from django.shortcuts import render, redirect
from django.contrib import messages
from .models import MLModel
from . import ml_utils
import pandas as pd
import numpy as np
import io
import os
import traceback
import logging
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Set up logging
logger = logging.getLogger(__name__)

def home(request):
    models = [
        {'id': 'slr', 'name': 'Health Score Predictor (SLR)', 'description': 'A simple model that predicts health score based on age and basic health metrics. Uses Simple Linear Regression.'},
        {'id': 'mlr', 'name': 'Advanced Health Predictor (MLR)', 'description': 'A comprehensive health score predictor using multiple health factors. Uses Multiple Linear Regression.'},
        {'id': 'poly', 'name': 'Advanced Health Predictor (Polynomial)', 'description': 'A non-linear health score predictor that captures complex relationships between health factors. Uses Polynomial Regression.'},
        {'id': 'logr', 'name': 'Health Risk Classifier (LogR)', 'description': 'Classifies your health risk level based on various health metrics. Uses Logistic Regression.'},
        {'id': 'knn', 'name': 'Health Group Classifier (KNN)', 'description': 'Classifies your health into similarity groups based on your health metrics. Uses K-Nearest Neighbors.'},
    ]
    
    return render(request, 'ml_models/home.html', {'models': models})

def run_model(request, model_id):
    """
    Run the specified machine learning model
    """
    
    model_mapping = {
        'slr': {'name': 'Health Score Predictor (SLR)', 'description': 'A simple model that predicts health score based on age and basic health metrics.'},
        'mlr': {'name': 'Advanced Health Predictor (MLR)', 'description': 'A comprehensive health score predictor using multiple health factors.'},
        'poly': {'name': 'Advanced Health Predictor (Polynomial)', 'description': 'A non-linear health score predictor that captures complex relationships between health factors.'},
        'logr': {'name': 'Health Risk Classifier (LogR)', 'description': 'Classifies your health risk level based on various health metrics.'},
        'knn': {'name': 'Health Group Classifier (KNN)', 'description': 'Classifies your health into similarity groups based on your health metrics.'},
    }
    
    if model_id not in model_mapping:
        return redirect('home')
    
    model_name = model_mapping[model_id]['name']
    model_description = model_mapping[model_id]['description']
    result = None
    form_data = {}
    
    if request.method == 'POST':
        # Initialize form data
        if model_name == 'Health Score Predictor (SLR)':
            form_data = {
                'age': request.POST.get('age', ''),
                'weight': request.POST.get('weight', ''),
                'height': request.POST.get('height', ''),
                'exercise_hours': request.POST.get('exercise_hours', '')
            }
            result = predict_health_with_slr(request)
        elif model_name == 'Advanced Health Predictor (MLR)':
            form_data = {
                'age': request.POST.get('age', ''),
                'weight': request.POST.get('weight', ''),
                'height': request.POST.get('height', ''),
                'exercise_hours': request.POST.get('exercise_hours', ''),
                'diet_quality': request.POST.get('diet_quality', ''),
                'sleep_hours': request.POST.get('sleep_hours', '')
            }
            result = predict_health_with_mlr(request)
        elif model_name == 'Advanced Health Predictor (Polynomial)':
            form_data = {
                'age': request.POST.get('age', ''),
                'polynomial_degree': request.POST.get('polynomial_degree', '2')
            }
            result = predict_health_with_poly(request)
        elif model_name == 'Health Risk Classifier (LogR)':
            form_data = {
                'age': request.POST.get('age', ''),
                'weight': request.POST.get('weight', ''),
                'height': request.POST.get('height', ''),
                'exercise_hours': request.POST.get('exercise_hours', ''),
                'systolic_bp': request.POST.get('systolic_bp', ''),
                'diastolic_bp': request.POST.get('diastolic_bp', ''),
                'cholesterol': request.POST.get('cholesterol', ''),
                'glucose': request.POST.get('glucose', ''),
                'smoking': request.POST.get('smoking', ''),
                'alcohol': request.POST.get('alcohol', '')
            }
            result = predict_health_with_logr(request)
        elif model_name == 'Health Group Classifier (KNN)':
            form_data = {
                'age': request.POST.get('age', ''),
                'weight': request.POST.get('weight', ''),
                'height': request.POST.get('height', ''),
                'exercise_hours': request.POST.get('exercise_hours', ''),
                'sleep_hours': request.POST.get('sleep_hours', ''),
                'diet_quality': request.POST.get('diet_quality', ''),
                'stress_level': request.POST.get('stress_level', ''),
                'n_neighbors': request.POST.get('n_neighbors', '5')
            }
            result = predict_health_with_knn(request)
        else:
            # Handle existing processing for other models
            form_data = {}
            # For example, process CSV data
            if 'input_data' in request.POST:
                input_data = request.POST.get('input_data', '')
                try:
                    if model_name == 'Polynomial Regression':
                        degree = int(request.POST.get('degree', 2))
                        result = process_input_data(model_name, input_data, degree=degree)
                    elif model_name == 'K-Nearest Neighbors':
                        n_neighbors = int(request.POST.get('neighbors', 3))
                        result = process_input_data(model_name, input_data, n_neighbors=n_neighbors)
                    else:
                        result = process_input_data(model_name, input_data)
                except Exception as e:
                    logging.error(f"Error running model {model_id}: {str(e)}")
                    result = {"error": str(e)}
            else:
                result = {"error": "No input data provided"}
    
    context = {
        'model_name': model_name,
        'model_description': model_description,
        'result': result,
        'form_data': form_data
    }
    
    return render(request, 'ml_models/model.html', context)

def validate_csv_data(input_data, model_name):
    """Validate CSV data format and required columns"""
    try:
        # Try to parse as CSV
        df = pd.read_csv(io.StringIO(input_data))
        
        # Check for required columns based on model type
        required_columns = {
            'linear_regression': ['x', 'y'],
            'multiple_regression': ['area', 'bedrooms', 'age', 'price'],
            'polynomial_regression': ['x', 'y'],
            'logistic_regression': ['hours_studied', 'previous_score', 'extracurricular', 'sleep_hours', 'passed'],
            'knn': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
            'blood_pressure': ['age', 'weight', 'height', 'gender', 'exercise_hours_per_week', 
                              'stress_level', 'sodium_intake', 'systolic', 'diastolic']
        }
        
        columns = required_columns.get(model_name, [])
        missing_columns = [col for col in columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
        # Check for empty dataframe
        if df.empty:
            raise ValueError("The input data is empty.")
            
        # Check for missing values
        if df.isnull().values.any():
            raise ValueError("The input data contains missing values.")
            
        # Additional validation for specific models
        if model_name == 'knn':
            if not all(df['species'].isin(['setosa', 'versicolor', 'virginica'])):
                raise ValueError("Species must be one of: setosa, versicolor, virginica")
                
        if model_name == 'logistic_regression':
            if not all(df['passed'].isin([0, 1])):
                raise ValueError("The 'passed' column must contain only 0 or 1 values")
                
        if model_name == 'blood_pressure':
            if not all(df['gender'].isin([0, 1])):
                raise ValueError("Gender must be encoded as 0 (female) or 1 (male)")
            
            # Validate numerical ranges for blood pressure
            if (df['systolic'] < 70).any() or (df['systolic'] > 200).any():
                raise ValueError("Systolic blood pressure values must be between 70 and 200 mmHg")
                
            if (df['diastolic'] < 40).any() or (df['diastolic'] > 120).any():
                raise ValueError("Diastolic blood pressure values must be between 40 and 120 mmHg")
                
        return True
        
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format. Please check your input data.")
    except Exception as e:
        raise ValueError(f"Data validation error: {str(e)}")

def load_default_data(model_name):
    """Load default data from file for display in the form"""
    try:
        file_path = ml_utils.get_model_data_path(model_name)
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading default data for {model_name}: {str(e)}")
        return ""

def process_input_data(model_name, input_data, **kwargs):
    """Process user input data and run the appropriate model"""
    result = {}
    
    try:
        # Convert input text to DataFrame
        df = pd.read_csv(io.StringIO(input_data))
        
        # Apply preprocessing based on the model type
        if model_name == 'linear_regression':
            X = df[['x']].values
            y = df['y'].values
            
        elif model_name == 'multiple_regression':
            X = df[['area', 'bedrooms', 'age']].values
            y = df['price'].values
            
            # Scale the features
            X = ml_utils.scale_features(X)
            
        elif model_name == 'polynomial_regression':
            X = df[['x']].values
            y = df['y'].values
            degree = kwargs.get('degree', 3)
            
        elif model_name == 'logistic_regression':
            X = df[['hours_studied', 'previous_score', 'extracurricular', 'sleep_hours']].values
            y = df['passed'].values
            
            # Scale features
            X = ml_utils.scale_features(X)
            
        elif model_name == 'knn':
            X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
            
            # Convert species to numerical values
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(df['species'])
        
            # Scale features
            X = ml_utils.scale_features(X)
            n_neighbors = kwargs.get('n_neighbors', 3)
            
        elif model_name == 'blood_pressure':
            # Extract features
            X = df[['age', 'weight', 'height', 'gender', 'exercise_hours_per_week', 
                   'stress_level', 'sodium_intake']].values
            
            # Extract both blood pressure values as targets
            y = df[['systolic', 'diastolic']].values
            
            # Scale features
            X = ml_utils.scale_features(X)
        
        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Process based on model type
        if model_name == 'linear_regression':
            # Train model
            model = ml_utils.train_linear_regression(X_train, y_train)
            
            # Evaluate model
            evaluation = ml_utils.evaluate_regression_model(model, X_test, y_test)
            
            result = {
                'mse': round(evaluation['mse'], 3),
                'r2': round(evaluation['r2'], 3)
            }
            
        elif model_name == 'multiple_regression':
            # Train model
            model = ml_utils.train_multiple_regression(X_train, y_train)
            
            # Evaluate model
            evaluation = ml_utils.evaluate_regression_model(model, X_test, y_test)
            
            # Get feature importance (coefficients)
            feature_names = ['Area', 'Bedrooms', 'Age']
            coef = model.coef_
            importance = {feature: round(coef[i], 3) for i, feature in enumerate(feature_names)}
            
            result = {
                'mse': round(evaluation['mse'], 3),
                'r2': round(evaluation['r2'], 3),
                'importance': importance
            }
            
        elif model_name == 'polynomial_regression':
            # Train model with the specified degree
            model, poly = ml_utils.train_polynomial_regression(X_train, y_train, degree=degree)
            
            # Evaluate model
            evaluation = ml_utils.evaluate_regression_model(
                model, X_test, y_test, model_type=model_name, poly=poly
            )
            
            result = {
                'mse': round(evaluation['mse'], 3),
                'r2': round(evaluation['r2'], 3),
                'degree': poly.degree
            }
            
        elif model_name == 'logistic_regression':
            # Train model
            model = ml_utils.train_logistic_regression(X_train, y_train)
            
            # Evaluate model
            evaluation = ml_utils.evaluate_classification_model(model, X_test, y_test)
            
            # Feature importance
            feature_names = ['Hours Studied', 'Previous Score', 'Extracurricular', 'Sleep Hours']
            coef = model.coef_[0]
            importance = {feature: round(coef[i], 3) for i, feature in enumerate(feature_names)}
            
            result = {
                'accuracy': round(evaluation['accuracy'], 3),
                'importance': importance
            }
            
        elif model_name == 'knn':
            # Train model with the specified number of neighbors
            model = ml_utils.train_knn(X_train, y_train, n_neighbors=n_neighbors)
            
            # Evaluate model
            evaluation = ml_utils.evaluate_classification_model(model, X_test, y_test)
            
            result = {
                'accuracy': round(evaluation['accuracy'], 3),
                'n_neighbors': model.n_neighbors
            }
            
        elif model_name == 'blood_pressure':
            # Train blood pressure model
            model, model_type = ml_utils.train_blood_pressure_model(X_train, y_train)
            
            # Evaluate model
            evaluation = ml_utils.evaluate_blood_pressure_model(model, X_test, y_test)
            
            # Get feature importance if available
            feature_names = ['Age', 'Weight', 'Height', 'Gender', 'Exercise', 'Stress', 'Sodium']
            importance = {}
            
            # Only linear models have meaningful coefficients
            if hasattr(model, 'estimators_') and model_type in ['linear', 'ridge', 'lasso']:
                for i, estimator in enumerate(['Systolic', 'Diastolic']):
                    est = model.estimators_[i]
                    if hasattr(est, 'coef_'):
                        coef = est.coef_
                        for j, feature in enumerate(feature_names):
                            key = f"{estimator} - {feature}"
                            importance[key] = round(float(coef[j]), 3)
            
            # Prepare rounded results
            result = {
                'mse': [round(val, 3) for val in evaluation['mse']],
                'mae': [round(val, 3) for val in evaluation['mae']],
                'r2': [round(val, 3) for val in evaluation['r2']],
                'mape': [round(val, 3) for val in evaluation['mape']],
                'model_type': model_type,
                'importance': importance if importance else None
            }
            
        return result
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"Error processing model: {str(e)}")

def predict_blood_pressure(age, weight, height, gender, exercise_hours, stress_level, sodium_intake):
    """Predict blood pressure from direct user input"""
    try:
        # Load the training data to train the model
        file_path = ml_utils.get_model_data_path('blood_pressure')
        df = pd.read_csv(file_path)
        
        # Prepare training data
        X_train = df[['age', 'weight', 'height', 'gender', 'exercise_hours_per_week', 
                    'stress_level', 'sodium_intake']].values
        y_train = df[['systolic', 'diastolic']].values
        
        # Scale training features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model, model_type = ml_utils.train_blood_pressure_model(X_train_scaled, y_train)
        
        # Prepare the user input for prediction
        user_input = np.array([[age, weight, height, gender, exercise_hours, stress_level, sodium_intake]])
        
        # Scale the user input using the same scaler
        user_input_scaled = scaler.transform(user_input)
        
        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        
        # Round to nearest integer for blood pressure values
        systolic = round(prediction[0])
        diastolic = round(prediction[1])
        
        # Calculate blood pressure category
        category = classify_blood_pressure(systolic, diastolic)
        
        # Get feature importance if available
        feature_names = ['Age', 'Weight', 'Height', 'Gender', 'Exercise', 'Stress', 'Sodium']
        importance = {}
        
        # Only linear models have meaningful coefficients
        if hasattr(model, 'estimators_') and model_type in ['linear', 'ridge', 'lasso']:
            for i, estimator in enumerate(['Systolic', 'Diastolic']):
                est = model.estimators_[i]
                if hasattr(est, 'coef_'):
                    coef = est.coef_
                    for j, feature in enumerate(feature_names):
                        key = f"{estimator} - {feature}"
                        importance[key] = round(float(coef[j]), 3)
        
        # Create result dictionary
        result = {
            'systolic': systolic,
            'diastolic': diastolic,
            'category': category,
            'model_type': model_type,
            'importance': importance if importance else None,
            'direct_prediction': True
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in blood pressure prediction: {str(e)}\n{traceback.format_exc()}")
        raise

def classify_blood_pressure(systolic, diastolic):
    """Classify blood pressure reading into a category"""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif (systolic >= 120 and systolic <= 129) and diastolic < 80:
        return "Elevated"
    elif (systolic >= 130 and systolic <= 139) or (diastolic >= 80 and diastolic <= 89):
        return "Hypertension Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2"
    elif systolic > 180 or diastolic > 120:
        return "Hypertensive Crisis (consult doctor immediately)"
    else:
        return "Unknown"

def process_linear_regression_input(x_values, y_values, prediction_input=None):
    """Process Simple Linear Regression with direct user input X-Y values"""
    try:
        # Convert to numpy arrays
        X = np.array(x_values).reshape(-1, 1)
        y = np.array(y_values)
        
        # Split data into training and testing sets
        # Use a smaller test size if we have few data points
        test_size = min(0.2, 1/len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train linear regression model
        model = ml_utils.train_linear_regression(X_train, y_train)
        
        # Evaluate model
        evaluation = ml_utils.evaluate_regression_model(model, X_test, y_test)
        
        # Calculate the slope and intercept (formula: y = mx + b)
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            # If using pipeline with named steps (from recent changes)
            regressor = model.named_steps['regressor']
            slope = regressor.coef_[0]
            intercept = regressor.intercept_
        else:
            # Direct access if not using pipeline
            slope = model.coef_[0] if hasattr(model, 'coef_') else None
            intercept = model.intercept_ if hasattr(model, 'intercept_') else None
        
        # Generate the equation string
        equation = f"y = {slope:.4f}x + {intercept:.4f}" if slope is not None and intercept is not None else "Unknown"
        
        # Make prediction if requested
        prediction = None
        if prediction_input is not None:
            prediction_x = np.array([[prediction_input]])
            prediction = float(model.predict(prediction_x)[0])
            
        # Create result dictionary
        result = {
            'mse': round(evaluation['mse'], 3),
            'r2': round(evaluation['r2'], 3),
            'equation': equation,
            'slope': round(slope, 4) if slope is not None else None,
            'intercept': round(intercept, 4) if intercept is not None else None,
            'data_points': len(x_values),
            'direct_input': True,
            'x_values': x_values,
            'y_values': y_values,
            'prediction_input': prediction_input,
            'prediction_result': round(prediction, 4) if prediction is not None else None
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in linear regression processing: {str(e)}\n{traceback.format_exc()}")
        raise ValueError(f"Error processing linear regression: {str(e)}")

def predict_health_with_slr(request):
    """Predict health outcome using Simple Linear Regression from direct user input"""
    try:
        # Get form data
        age = float(request.POST.get('age', 0))
        
        # Validate input
        if age < 18 or age > 100:
            return {"error": "Age must be between 18 and 100 years."}
        
        # Create a feature array from user input
        user_input = np.array([[age]])
        
        # Create some synthetic training data
        ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).reshape(-1, 1)
        health_scores = np.array([90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70])
        
        # Train the model
        model = ml_utils.train_linear_regression(ages, health_scores)
        
        # Make prediction
        prediction = float(model.predict(user_input)[0])
        
        # Calculate the slope and intercept
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            regressor = model.named_steps['regressor']
            slope = regressor.coef_[0]
            intercept = regressor.intercept_
        else:
            slope = model.coef_[0] if hasattr(model, 'coef_') else None
            intercept = model.intercept_ if hasattr(model, 'intercept_') else None
        
        # Generate the equation string
        equation = f"Health Score = {slope:.4f} × Age + {intercept:.4f}" if slope is not None and intercept is not None else "Unknown"
        
        # Classify health score
        category = classify_health_score(prediction)
        
        # Create result dictionary
        result = {
            'health_score': round(prediction, 1),
            'equation': equation,
            'category': category,
            'age': age,
            'direct_input': True
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in health prediction: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"An error occurred: {str(e)}"}

def classify_health_score(score):
    """Classify health score into a category"""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Very Good"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Poor"

def predict_health_with_mlr(request):
    """Predict health outcome using Multiple Linear Regression from direct user input"""
    try:
        # Get form data
        age = float(request.POST.get('age', 0))
        weight = float(request.POST.get('weight', 0))
        height = float(request.POST.get('height', 0))
        exercise_hours = float(request.POST.get('exercise_hours', 0))
        diet_quality = float(request.POST.get('diet_quality', 0))
        sleep_hours = float(request.POST.get('sleep_hours', 0))
        
        # Validate inputs
        if age < 18 or age > 100:
            return {"error": "Age must be between 18 and 100 years."}
        if weight < 40 or weight > 200:
            return {"error": "Weight must be between 40 and 200 kg."}
        if height < 120 or height > 220:
            return {"error": "Height must be between 120 and 220 cm."}
        if exercise_hours < 0 or exercise_hours > 20:
            return {"error": "Exercise hours must be between 0 and 20 hours per week."}
        if diet_quality < 1 or diet_quality > 10:
            return {"error": "Diet quality must be between 1 and 10."}
        if sleep_hours < 4 or sleep_hours > 12:
            return {"error": "Sleep hours must be between 4 and 12 hours per day."}
        
        # Calculate BMI
        height_in_meters = height / 100
        bmi = weight / (height_in_meters ** 2)
        
        # Create a feature array from user inputs (using all available features)
        user_input = np.array([[age, bmi, exercise_hours, diet_quality, sleep_hours]])
        
        # Create synthetic training data with multiple features
        # In a real app, you would load this from a dataset
        # Features: age, bmi, exercise_hours, diet_quality, sleep_hours
        features = np.array([
            [20, 22.0, 5, 7, 8],  # Young, healthy lifestyle
            [25, 23.5, 4, 6, 7],
            [30, 24.0, 3, 6, 7], 
            [35, 25.0, 3, 5, 7],
            [40, 26.0, 2, 5, 6.5],
            [45, 26.5, 2, 4, 6.5],
            [50, 27.0, 1.5, 4, 6],
            [55, 27.5, 1.5, 3, 6],
            [60, 28.0, 1, 3, 6],
            [65, 28.5, 1, 3, 5.5],
            [70, 29.0, 0.5, 2, 5.5]  # Older, less active lifestyle
        ])
        
        # Target: health scores
        health_scores = np.array([95, 92, 88, 85, 82, 79, 76, 73, 70, 67, 64])
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(features)
        user_input_scaled = scaler.transform(user_input)
        
        # Train the model
        model = ml_utils.train_multiple_regression(X_train_scaled, health_scores)
        
        # Make prediction
        prediction = float(model.predict(user_input_scaled)[0])
        
        # Get feature importance
        feature_names = ['Age', 'BMI', 'Exercise', 'Diet', 'Sleep']
        coef = model.coef_
        importance = {feature: round(coef[i], 3) for i, feature in enumerate(feature_names)}
        
        # Classify health score
        category = classify_health_score(prediction)
        
        # Create result dictionary
        result = {
            'health_score': round(prediction, 1),
            'category': category,
            'age': age,
            'weight': weight,
            'height': height,
            'bmi': round(bmi, 1),
            'exercise_hours': exercise_hours,
            'diet_quality': diet_quality,
            'sleep_hours': sleep_hours,
            'importance': importance,
            'direct_input': True,
            'advanced_predictor': True
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in MLR health prediction: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"An error occurred: {str(e)}"}

def predict_health_with_poly(request):
    """
    Predict health score using Polynomial Regression based on age
    """
    try:
        # Get form data
        age = float(request.POST.get('age', 0))
        polynomial_degree = int(request.POST.get('polynomial_degree', 2))
        
        # Validate input
        if age < 18 or age > 100:
            return {"error": "Age must be between 18 and 100 years."}
        if polynomial_degree < 1 or polynomial_degree > 5:
            return {"error": "Polynomial degree must be between 1 and 5."}
        
        # Create a feature array from user input
        user_input = np.array([[age]])
        
        # Create synthetic training data
        ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).reshape(-1, 1)
        health_scores = np.array([90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70])
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=polynomial_degree)
        X_train_poly = poly.fit_transform(ages)
        user_input_poly = poly.transform(user_input)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train_poly, health_scores)
        
        # Make prediction
        prediction = float(model.predict(user_input_poly)[0])
        
        # Get coefficients for equation
        coef = model.coef_
        intercept = model.intercept_
        
        # Generate equation string
        equation_terms = []
        for i, c in enumerate(coef[1:]):  # Skip the constant term
            if abs(c) > 1e-10:  # Only include non-zero terms
                term = f"{c:.2f}*Age^{i+1}"
                if i == 0 or c < 0:
                    equation_terms.append(term)
                else:
                    equation_terms.append(f"+ {term}")
        
        equation = f"Health Score = {intercept:.4f} {' '.join(equation_terms)}"
        
        # Classify health score
        category = classify_health_score(prediction)
        
        # Create result dictionary
        result = {
            'health_score': round(prediction, 1),
            'equation': equation,
            'category': category,
            'age': age,
            'polynomial_degree': polynomial_degree,
            'direct_input': True
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in polynomial health prediction: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"An error occurred: {str(e)}"}

def predict_health_with_logr(request):
    """
    Predict health risk category using Logistic Regression based on health metrics
    """
    try:
        # Get form data
        age = float(request.POST.get('age', 0))
        weight = float(request.POST.get('weight', 0))
        height = float(request.POST.get('height', 0))
        exercise_hours = float(request.POST.get('exercise_hours', 0))
        systolic_bp = float(request.POST.get('systolic_bp', 0))
        diastolic_bp = float(request.POST.get('diastolic_bp', 0))
        cholesterol = float(request.POST.get('cholesterol', 0))
        glucose = float(request.POST.get('glucose', 0))
        smoking = int(request.POST.get('smoking', 0))
        alcohol = int(request.POST.get('alcohol', 0))
        
        # Validate inputs
        if age < 18 or age > 100:
            return {"error": "Age must be between 18 and 100 years."}
        if weight < 40 or weight > 200:
            return {"error": "Weight must be between 40 and 200 kg."}
        if height < 120 or height > 220:
            return {"error": "Height must be between 120 and 220 cm."}
        if exercise_hours < 0 or exercise_hours > 20:
            return {"error": "Exercise hours must be between 0 and 20 hours per week."}
        if systolic_bp < 80 or systolic_bp > 220:
            return {"error": "Systolic blood pressure must be between 80 and 220 mmHg."}
        if diastolic_bp < 40 or diastolic_bp > 140:
            return {"error": "Diastolic blood pressure must be between 40 and 140 mmHg."}
        if cholesterol < 100 or cholesterol > 400:
            return {"error": "Total cholesterol must be between 100 and 400 mg/dL."}
        if glucose < 50 or glucose > 200:
            return {"error": "Fasting glucose must be between 50 and 200 mg/dL."}
        if smoking not in [0, 1, 2, 3]:
            return {"error": "Smoking status must be between 0 and 3."}
        if alcohol not in [0, 1, 2, 3]:
            return {"error": "Alcohol consumption must be between 0 and 3."}
        
        # Calculate BMI
        height_m = height / 100
        bmi = round(weight / (height_m * height_m), 1)
        
        # Create a synthetic dataset for training the model
        np.random.seed(42)  # For reproducibility
        
        # Generate a larger synthetic dataset
        n_samples = 1000
        
        # Generate synthetic data with more realistic distributions
        synthetic_ages = np.random.normal(45, 15, n_samples).clip(18, 100)
        synthetic_bmis = np.random.normal(25, 5, n_samples).clip(15, 45)
        synthetic_exercise = np.random.gamma(shape=2, scale=1.5, size=n_samples).clip(0, 20)
        synthetic_systolic = np.random.normal(120, 15, n_samples).clip(80, 220)
        synthetic_diastolic = np.random.normal(80, 10, n_samples).clip(40, 140)
        synthetic_cholesterol = np.random.normal(190, 35, n_samples).clip(100, 400)
        synthetic_glucose = np.random.normal(95, 20, n_samples).clip(50, 200)
        synthetic_smoking = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.6, 0.2, 0.1, 0.1])
        synthetic_alcohol = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Create feature matrix
        X = np.column_stack([
            synthetic_ages,
            synthetic_bmis,
            synthetic_exercise,
            synthetic_systolic,
            synthetic_diastolic,
            synthetic_cholesterol,
            synthetic_glucose,
            synthetic_smoking,
            synthetic_alcohol
        ])
        
        # Define risk factors for health risk
        risk_factors = (
            (synthetic_ages > 55) * 1 +
            (synthetic_bmis > 30) * 1.5 +
            (synthetic_bmis < 18.5) * 1 +
            (synthetic_exercise < 2) * 1 +
            (synthetic_systolic > 140) * 1.5 +
            (synthetic_diastolic > 90) * 1.5 +
            (synthetic_cholesterol > 240) * 1.5 +
            (synthetic_glucose > 125) * 2 +
            (synthetic_smoking) * 0.8 +
            (synthetic_alcohol > 1) * 0.7
        )
        
        # Convert risk factors to risk categories (0: Low, 1: Moderate, 2: High)
        risk_categories = np.zeros(n_samples, dtype=int)
        risk_categories[(risk_factors >= 2) & (risk_factors < 4)] = 1  # Moderate risk
        risk_categories[risk_factors >= 4] = 2  # High risk
        
        # Create and train logistic regression model
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        # Create a pipeline with scaling and logistic regression
        logreg_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=0.8))
        ])
        
        # Train the model
        logreg_pipeline.fit(X, risk_categories)
        
        # Get feature importances (from the coefficients)
        coefficients = logreg_pipeline.named_steps['classifier'].coef_
        
        # For multinomial, there's a set of coefficients for each class
        # We'll use the average absolute value of coefficients across classes as importance
        feature_importance = np.mean(np.abs(coefficients), axis=0)
        
        # Normalize feature importances
        feature_importance = feature_importance / np.sum(feature_importance) * 10
        
        # Feature names
        feature_names = ['Age', 'BMI', 'Exercise', 'Systolic BP', 'Diastolic BP', 
                         'Cholesterol', 'Glucose', 'Smoking', 'Alcohol']
        
        # Create dictionary of feature importances
        importance = {name: round(imp, 2) for name, imp in zip(feature_names, feature_importance)}
        
        # Prepare user data for prediction
        user_data = np.array([[
            age, 
            bmi, 
            exercise_hours, 
            systolic_bp, 
            diastolic_bp, 
            cholesterol, 
            glucose, 
            smoking, 
            alcohol
        ]])
        
        # Make prediction
        risk_category_idx = logreg_pipeline.predict(user_data)[0]
        risk_probabilities = logreg_pipeline.predict_proba(user_data)[0]
        
        # Map index to risk category
        risk_categories_map = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk"
        }
        
        risk_category = risk_categories_map[risk_category_idx]
        
        # Round probabilities
        risk_probabilities = [round(p * 100, 1) for p in risk_probabilities]
        
        # Generate health recommendations based on risk factors
        recommendations = []
        
        if bmi > 30:
            recommendations.append("Weight management: Your BMI indicates obesity. Consider a balanced diet and regular exercise.")
        elif bmi > 25:
            recommendations.append("Weight management: Your BMI indicates overweight. Consider a balanced diet and regular exercise.")
        elif bmi < 18.5:
            recommendations.append("Weight management: Your BMI indicates underweight. Consider nutritional counseling.")
            
        if exercise_hours < 2:
            recommendations.append("Physical activity: Aim for at least 150 minutes of moderate exercise per week.")
            
        if systolic_bp >= 140 or diastolic_bp >= 90:
            recommendations.append("Blood pressure: Your blood pressure is elevated. Consider dietary changes, stress reduction, and consult a healthcare provider.")
            
        if cholesterol > 240:
            recommendations.append("Cholesterol management: Your cholesterol is high. Consider dietary changes and consult a healthcare provider.")
            
        if glucose > 125:
            recommendations.append("Blood glucose: Your glucose level is elevated. Consider dietary changes and consult a healthcare provider.")
            
        if smoking > 0:
            recommendations.append("Smoking cessation: Consider quitting smoking or reducing tobacco use.")
            
        if alcohol > 1:
            recommendations.append("Alcohol consumption: Consider reducing alcohol intake.")
            
        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations.append("Maintain a balanced diet rich in fruits, vegetables, and whole grains.")
            recommendations.append("Continue your current exercise routine.")
            recommendations.append("Get regular health check-ups.")
        
        # Add standard recommendations
        recommendations.append("Manage stress through relaxation techniques, mindfulness, or meditation.")
        recommendations.append("Get 7-9 hours of quality sleep per night.")
        
        return {
            'risk_category': risk_category,
            'risk_probabilities': {
                'Low Risk': risk_probabilities[0],
                'Moderate Risk': risk_probabilities[1],
                'High Risk': risk_probabilities[2]
            },
            'age': age,
            'bmi': bmi,
            'exercise_hours': exercise_hours,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'smoking': smoking,
            'alcohol': alcohol,
            'importance': importance,
            'recommendations': recommendations
        }
    
    except ValueError as e:
        logging.error(f"ValueError in health risk prediction: {str(e)}")
        return {"error": "Please ensure all fields contain valid numeric values."}
    except Exception as e:
        logging.error(f"Error in health risk prediction: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

def predict_health_with_knn(request):
    """
    Predict health group using K-Nearest Neighbors based on health metrics
    """
    try:
        # Get form data
        age = float(request.POST.get('age', 0))
        weight = float(request.POST.get('weight', 0))
        height = float(request.POST.get('height', 0))
        exercise_hours = float(request.POST.get('exercise_hours', 0))
        sleep_hours = float(request.POST.get('sleep_hours', 0))
        diet_quality = float(request.POST.get('diet_quality', 0))
        stress_level = float(request.POST.get('stress_level', 0))
        n_neighbors = int(request.POST.get('n_neighbors', 5))
        
        # Validate inputs
        if age < 18 or age > 100:
            return {"error": "Age must be between 18 and 100 years."}
        if weight < 40 or weight > 200:
            return {"error": "Weight must be between 40 and 200 kg."}
        if height < 120 or height > 220:
            return {"error": "Height must be between 120 and 220 cm."}
        if exercise_hours < 0 or exercise_hours > 20:
            return {"error": "Exercise hours must be between 0 and 20 hours per week."}
        if sleep_hours < 3 or sleep_hours > 12:
            return {"error": "Sleep hours must be between 3 and 12 hours per day."}
        if diet_quality < 1 or diet_quality > 10:
            return {"error": "Diet quality must be between 1 and 10."}
        if stress_level < 1 or stress_level > 10:
            return {"error": "Stress level must be between 1 and 10."}
        if n_neighbors < 1 or n_neighbors > 20:
            return {"error": "Number of neighbors must be between 1 and 20."}
        
        # Calculate BMI
        height_m = height / 100
        bmi = round(weight / (height_m * height_m), 1)
        
        # Create a synthetic dataset for training the model
        np.random.seed(42)  # For reproducibility
        
        # Generate a larger synthetic dataset
        n_samples = 500
        
        # Generate synthetic data with more realistic distributions
        synthetic_ages = np.random.normal(45, 15, n_samples).clip(18, 100)
        synthetic_bmis = np.random.normal(25, 5, n_samples).clip(15, 45)
        synthetic_exercise = np.random.gamma(shape=2, scale=1.5, size=n_samples).clip(0, 20)
        synthetic_sleep = np.random.normal(7, 1.5, n_samples).clip(3, 12)
        synthetic_diet = np.random.normal(6, 2, n_samples).clip(1, 10)
        synthetic_stress = np.random.normal(5, 2, n_samples).clip(1, 10)
        
        # Create feature matrix
        X = np.column_stack([
            synthetic_ages,
            synthetic_bmis,
            synthetic_exercise,
            synthetic_sleep, 
            synthetic_diet,
            synthetic_stress
        ])
        
        # Define health groups based on combinations of factors
        # Group 0: Optimal Health - Good metrics across the board
        # Group 1: Good Health with Some Concerns - Generally good but some issues
        # Group 2: Moderate Health Concerns - Multiple suboptimal metrics
        # Group 3: Significant Health Concerns - Poor metrics across multiple dimensions
        
        health_groups = np.zeros(n_samples, dtype=int)
        
        # Create a score for each sample based on their metrics
        health_scores = (
            (synthetic_bmis < 25) * 20 +  # Normal BMI
            (synthetic_bmis >= 25) * -10 +  # Overweight penalty
            (synthetic_bmis >= 30) * -10 +  # Additional obesity penalty
            (synthetic_bmis < 18.5) * -15 +  # Underweight penalty
            synthetic_exercise * 3 +  # Exercise bonus (up to 60 points)
            (synthetic_sleep >= 7) * 15 +  # Good sleep bonus
            (synthetic_sleep < 6) * -10 +  # Poor sleep penalty
            synthetic_diet * 3 +  # Diet quality (up to 30 points)
            (10 - synthetic_stress) * 3  # Lower stress is better (up to 27 points)
        )
        
        # Assign groups based on health scores
        health_groups[(health_scores >= 70) & (health_scores < 90)] = 1  # Good Health with Some Concerns
        health_groups[(health_scores >= 40) & (health_scores < 70)] = 2  # Moderate Health Concerns
        health_groups[health_scores < 40] = 3  # Significant Health Concerns
        
        # Create and train KNN model
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        
        # Create a pipeline with scaling and KNN
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])
        
        # Train the model
        knn_pipeline.fit(X, health_groups)
        
        # Calculate feature importances based on variance
        # KNN doesn't have built-in feature importance, so we use a simple heuristic
        scaler = knn_pipeline.named_steps['scaler']
        feature_variances = np.var(scaler.transform(X), axis=0)
        feature_importance = feature_variances / np.sum(feature_variances) * 10
        
        # Feature names
        feature_names = ['Age', 'BMI', 'Exercise', 'Sleep', 'Diet', 'Stress']
        
        # Create dictionary of feature importances
        importance = {name: round(imp, 2) for name, imp in zip(feature_names, feature_importance)}
        
        # Prepare user data for prediction
        user_data = np.array([[
            age, 
            bmi, 
            exercise_hours, 
            sleep_hours,
            diet_quality,
            stress_level
        ]])
        
        # Make prediction
        health_group = knn_pipeline.predict(user_data)[0]
        
        # Get the nearest neighbors
        neighbors = knn_pipeline.named_steps['classifier'].kneighbors(
            knn_pipeline.named_steps['scaler'].transform(user_data), 
            return_distance=True
        )
        
        neighbor_distances = neighbors[0][0]
        neighbor_indices = neighbors[1][0]
        
        # Calculate confidence based on distance to neighbors
        # Closer neighbors = higher confidence
        max_distance = np.max(neighbor_distances)
        confidences = 1 - (neighbor_distances / (max_distance * 2))
        avg_confidence = round(np.mean(confidences) * 100, 1)
        
        # Map index to health group
        health_groups_map = {
            0: "Optimal Health",
            1: "Good Health with Some Concerns",
            2: "Moderate Health Concerns",
            3: "Significant Health Concerns"
        }
        
        health_category = health_groups_map[health_group]
        
        # Generate health recommendations based on user metrics
        recommendations = []
        
        if bmi > 30:
            recommendations.append("Weight management: Your BMI indicates obesity. Consider a balanced diet and regular exercise.")
        elif bmi > 25:
            recommendations.append("Weight management: Your BMI indicates overweight. Consider a balanced diet and regular exercise.")
        elif bmi < 18.5:
            recommendations.append("Weight management: Your BMI indicates underweight. Consider nutritional counseling.")
            
        if exercise_hours < 2:
            recommendations.append("Physical activity: Aim for at least 150 minutes of moderate exercise per week.")
            
        if sleep_hours < 7:
            recommendations.append("Sleep improvement: Aim for 7-9 hours of quality sleep per night for optimal health.")
            
        if diet_quality < 5:
            recommendations.append("Diet improvement: Consider increasing consumption of fruits, vegetables, whole grains, and lean proteins.")
            
        if stress_level > 7:
            recommendations.append("Stress management: High stress levels can impact both mental and physical health. Consider relaxation techniques, mindfulness, or meditation.")
            
        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations.append("Maintain your current healthy lifestyle.")
            recommendations.append("Continue with your balanced diet, regular exercise, and good sleep habits.")
            recommendations.append("Consider preventive health screenings appropriate for your age.")
        
        # Add standard recommendations
        recommendations.append("Regular health check-ups are important for preventive care.")
        
        # Create a list of similar profiles (the neighbors)
        neighbor_profiles = []
        for i, idx in enumerate(neighbor_indices):
            similarity = round(confidences[i] * 100, 1)
            if similarity > 20:  # Only include reasonably similar profiles
                neighbor_profiles.append({
                    'similarity': similarity,
                    'age': round(synthetic_ages[idx]),
                    'bmi': round(synthetic_bmis[idx], 1),
                    'exercise': round(synthetic_exercise[idx], 1),
                    'sleep': round(synthetic_sleep[idx], 1),
                    'diet': round(synthetic_diet[idx]),
                    'stress': round(synthetic_stress[idx]),
                    'group': health_groups_map[health_groups[idx]]
                })
        
        return {
            'health_category': health_category,
            'confidence': avg_confidence,
            'age': age,
            'bmi': bmi,
            'exercise_hours': exercise_hours,
            'sleep_hours': sleep_hours,
            'diet_quality': diet_quality,
            'stress_level': stress_level,
            'importance': importance,
            'recommendations': recommendations,
            'neighbor_profiles': neighbor_profiles,
            'n_neighbors': n_neighbors
        }
    
    except ValueError as e:
        logging.error(f"ValueError in health group prediction: {str(e)}")
        return {"error": "Please ensure all fields contain valid numeric values."}
    except Exception as e:
        logging.error(f"Error in health group prediction: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}
