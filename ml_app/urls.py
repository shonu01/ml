from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('model/<str:model_id>/', views.run_model, name='run_model'),  # Model page
]
