"""
app/admin.py
Django admin configuration for FreshScan AI.
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import FoodScan


@admin.register(FoodScan)
class FoodScanAdmin(admin.ModelAdmin):
    list_display = [
        'food_name', 'prediction_badge', 'freshness_score',
        'confidence_score', 'shelf_life_days', 'expiry_date', 'timestamp'
    ]
    list_filter = ['prediction', 'food_category', 'timestamp']
    search_fields = ['food_name', 'prediction']
    readonly_fields = ['timestamp', 'image_preview']
    ordering = ['-timestamp']

    def prediction_badge(self, obj):
        color_map = {
            'Fresh': '#00C853',
            'Semi-Fresh': '#FFB300',
            'Rotten': '#D32F2F',
        }
        color = color_map.get(obj.prediction, '#grey')
        return format_html(
            '<span style="background:{};color:white;padding:3px 10px;border-radius:12px;">{}</span>',
            color, obj.prediction
        )
    prediction_badge.short_description = 'Prediction'

    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="200" />', obj.image.url)
        return "No image"
    image_preview.short_description = 'Image Preview'