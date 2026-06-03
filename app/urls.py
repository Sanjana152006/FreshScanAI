"""
app/urls.py
URL patterns for the FreshScan AI app.
"""

from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('result/<int:scan_id>/', views.result, name='result'),
    path('history/', views.history, name='history'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('about/', views.about, name='about'),

    # Reports & downloads
    path('download-pdf/<int:scan_id>/', views.download_pdf, name='download_pdf'),
    path('generate-qr/<int:scan_id>/', views.generate_qr, name='generate_qr'),

    # Webcam
    path('webcam-predict/', views.webcam_predict, name='webcam_predict'),

    # Delete
    path('delete/<int:scan_id>/', views.delete_scan, name='delete_scan'),
]