# Generated by Django 5.0.2 on 2025-03-27 03:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MLModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(choices=[('linear_regression', 'Simple Linear Regression'), ('multiple_regression', 'Multiple Linear Regression'), ('polynomial_regression', 'Polynomial Regression'), ('logistic_regression', 'Logistic Regression'), ('knn', 'K-Nearest Neighbors')], max_length=50)),
                ('description', models.TextField()),
            ],
        ),
    ]
