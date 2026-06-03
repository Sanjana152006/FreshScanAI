"""
app/models.py
Database models for FreshScan AI.
Stores all scan predictions in SQLite.
"""

from django.db import models
from django.utils import timezone


class FoodScan(models.Model):
    """
    Model to store each food freshness scan result.
    """

    FRESHNESS_CHOICES = [
        ('Fresh', 'Fresh'),
        ('Semi-Fresh', 'Semi-Fresh'),
        ('Rotten', 'Rotten'),
    ]

    FOOD_CATEGORY_CHOICES = [
        ('fruit', 'Fruit'),
        ('vegetable', 'Vegetable'),
        ('dairy', 'Dairy'),
        ('meat', 'Meat'),
        ('other', 'Other'),
    ]

    # Image upload
    image = models.ImageField(upload_to='uploads/', verbose_name="Food Image")

    # Food details
    food_name = models.CharField(max_length=200, verbose_name="Detected Food Name")
    food_category = models.CharField(
        max_length=50,
        choices=FOOD_CATEGORY_CHOICES,
        default='other',
        verbose_name="Food Category"
    )

    # Prediction results
    prediction = models.CharField(
        max_length=20,
        choices=FRESHNESS_CHOICES,
        verbose_name="Freshness Prediction"
    )
    freshness_score = models.FloatField(
        default=0.0,
        verbose_name="Freshness Score (%)"
    )
    confidence_score = models.FloatField(
        default=0.0,
        verbose_name="Confidence Score (%)"
    )

    # Shelf life prediction
    shelf_life_days = models.IntegerField(
        default=0,
        verbose_name="Shelf Life Remaining (days)"
    )
    shelf_life_hours = models.IntegerField(
        default=0,
        verbose_name="Shelf Life Remaining (hours)"
    )

    # Expiry date
    expiry_date = models.DateField(
        null=True,
        blank=True,
        verbose_name="Estimated Expiry Date"
    )

    # AI storage suggestion
    storage_suggestion = models.TextField(
        blank=True,
        verbose_name="AI Storage Suggestion"
    )

    # Timestamp
    timestamp = models.DateTimeField(
        default=timezone.now,
        verbose_name="Scan Timestamp"
    )

    # User agent (optional)
    user_ip = models.GenericIPAddressField(
        null=True,
        blank=True,
        verbose_name="User IP"
    )

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Food Scan"
        verbose_name_plural = "Food Scans"

    def __str__(self):
        return f"{self.food_name} - {self.prediction} ({self.freshness_score:.1f}%) - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    def get_freshness_color(self):
        """Return Bootstrap color class based on prediction."""
        if self.prediction == 'Fresh':
            return 'success'
        elif self.prediction == 'Semi-Fresh':
            return 'warning'
        else:
            return 'danger'

    def get_freshness_emoji(self):
        """Return emoji based on prediction."""
        if self.prediction == 'Fresh':
            return '✅'
        elif self.prediction == 'Semi-Fresh':
            return '⚠️'
        else:
            return '❌'