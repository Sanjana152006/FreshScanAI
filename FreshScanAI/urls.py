"""
FreshScanAI/urls.py
Main URL configuration for FreshScanAI project.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.urls')),  # Include app URLs
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)