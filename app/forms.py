"""
app/forms.py
Django forms for image upload.
"""

from django import forms


class FoodImageUploadForm(forms.Form):
    """
    Form for uploading a food image for freshness detection.
    """
    image = forms.ImageField(
        label='Upload Food Image',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/jpeg,image/png,image/jpg',
            'id': 'foodImageInput',
        }),
        help_text='Supported formats: JPG, PNG, JPEG. Max size: 10MB'
    )