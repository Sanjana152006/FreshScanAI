# app/apps.py
# Django app configuration for FreshScan AI.

from django.apps import AppConfig


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    verbose_name = 'FreshScan AI'