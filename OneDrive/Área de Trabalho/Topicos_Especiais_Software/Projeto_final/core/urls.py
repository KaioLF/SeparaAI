from django.urls import path
from .views import index, upload_file, data_analysis, train_model, predict_view, data_analysis, add_record_view

urlpatterns = [
    path('', index),
    path('predictor/', train_model, name='predictor'),
    path('upload/', upload_file, name='upload_file'),
    path('data_analysis/', data_analysis, name='data_analysis'),
    path('predict/', predict_view, name='predict_view'),
    path('add_record/', add_record_view, name='add_record'),
]
