from django.db import models

# Create your models here.

class MLModel(models.Model):
    MODEL_CHOICES = [
        ('linear_regression', 'Simple Linear Regression'),
        ('multiple_regression', 'Multiple Linear Regression'),
        ('polynomial_regression', 'Polynomial Regression'),
        ('logistic_regression', 'Logistic Regression'),
        ('knn', 'K-Nearest Neighbors'),
        ('blood_pressure', 'Blood Pressure Prediction'),
    ]
    
    name = models.CharField(max_length=50, choices=MODEL_CHOICES)
    description = models.TextField()
    
    @classmethod
    def get_model_description(cls, model_name):
        descriptions = {
            'linear_regression': 'Simple Linear Regression models the relationship between a dependent variable and one independent variable using a linear equation. It\'s used for predicting continuous values.',
            'multiple_regression': 'Multiple Linear Regression extends simple linear regression to include multiple independent variables. It\'s used when there are several factors influencing the outcome.',
            'polynomial_regression': 'Polynomial Regression fits a non-linear relationship between the independent and dependent variables by expressing the dependent variable as an nth degree polynomial. It captures curvilinear relationships in the data.',
            'logistic_regression': 'Logistic Regression is used for binary classification problems. It predicts the probability of an outcome that can only have two values (e.g., pass/fail, win/lose, healthy/sick).',
            'knn': 'K-Nearest Neighbors is a simple, non-parametric classification algorithm that classifies new data points based on the majority class of their k nearest neighbors in the feature space.',
            'blood_pressure': 'Blood Pressure Prediction uses multiple regression techniques to predict both systolic and diastolic blood pressure values based on health metrics like age, weight, height, exercise habits, and other lifestyle factors.',
        }
        return descriptions.get(model_name, 'No description available')
