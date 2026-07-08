from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('api/checkpoints/', views.api_checkpoints, name='api_checkpoints'),
    path('api/elo/', views.api_elo, name='api_elo'),
]