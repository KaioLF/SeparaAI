from django.urls import path
from .views import index, predictor, upload_file, data_analysis, train_model, predict_view

urlpatterns = [
    path('', index),
    path('predictor/', train_model, name='predictor'),
    path('upload/', upload_file, name='upload_file'),
    path('data_analysis/<str:file_path>/', data_analysis, name='data_analysis'),
    path('predict/', predict_view, name='predict_view'),
    # demais rotas
]
