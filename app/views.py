"""
app/views.py
Main views for FreshScan AI.
Handles image upload, prediction, history, dashboard, and report generation.
"""

import os
import io
import json
import base64
import qrcode
import pyttsx3
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.utils import timezone
from django.db.models import Count, Avg
from django.views.decorators.csrf import csrf_exempt

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.units import inch

from .models import FoodScan
from .forms import FoodImageUploadForm

# ─────────────────────────────────────────────
# Load TensorFlow model (lazy load to save memory)
# ─────────────────────────────────────────────
_model = None

def get_model():
    """Lazy-load the TensorFlow model."""
    global _model
    if _model is None:
        try:
            import tensorflow as tf
            model_path = settings.MODEL_PATH
            if os.path.exists(model_path):
                _model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded from {model_path}")
            else:
                print(f"⚠️  Model not found at {model_path}. Using demo mode.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    return _model


# ─────────────────────────────────────────────
# Food knowledge base for storage suggestions
# ─────────────────────────────────────────────
FOOD_SUGGESTIONS = {
    'Fresh': {
        'default': 'Store in a cool, dry place. Refrigerate if possible to extend shelf life.',
        'fruit': 'Keep refrigerated at 35–40°F. Place in produce drawer. Wash before eating.',
        'vegetable': 'Store in refrigerator crisper drawer. Keep away from ethylene-producing fruits.',
        'dairy': 'Keep refrigerated below 40°F. Consume before expiry date.',
        'meat': 'Refrigerate immediately. Use within 2–3 days or freeze for longer storage.',
    },
    'Semi-Fresh': {
        'default': 'Consume within 24–48 hours. Refrigerate immediately. Check for spoilage signs.',
        'fruit': 'Refrigerate immediately. Consume within 1–2 days. Good for smoothies/cooking.',
        'vegetable': 'Refrigerate and use within 24 hours. Great for cooking or blanching.',
        'dairy': 'Use immediately. Do not refrigerate again if already opened.',
        'meat': '⚠️ Cook immediately or discard. Do not store further.',
    },
    'Rotten': {
        'default': '❌ UNSAFE FOR CONSUMPTION. Dispose of immediately. Do not eat.',
        'fruit': '❌ Discard immediately. May contain harmful molds or bacteria.',
        'vegetable': '❌ Do not consume. Dispose in compost or trash.',
        'dairy': '❌ Discard immediately. Risk of food poisoning.',
        'meat': '❌ DANGER: Discard immediately. Risk of serious illness.',
    }
}

SHELF_LIFE_MAP = {
    'Fresh': {'days': 5, 'hours': 12},
    'Semi-Fresh': {'days': 1, 'hours': 8},
    'Rotten': {'days': 0, 'hours': 0},
}

# Class labels from trained model
CLASS_LABELS = [
    'fresh_fruits',
    'fresh_vegetables',
    'rotten_fruits',
    'rotten_vegetables',
]

# Food name detection keywords (simple heuristic)
FOOD_NAMES = {
    'apple': 'Apple', 'banana': 'Banana', 'mango': 'Mango',
    'tomato': 'Tomato', 'carrot': 'Carrot', 'broccoli': 'Broccoli',
    'orange': 'Orange', 'grape': 'Grape', 'strawberry': 'Strawberry',
    'potato': 'Potato', 'onion': 'Onion', 'spinach': 'Spinach',
    'lemon': 'Lemon', 'watermelon': 'Watermelon', 'pineapple': 'Pineapple',
}


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image using OpenCV for model input."""
    img = cv2.imread(str(image_path))
    if img is None:
        # Fallback to PIL
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize
    img_resized = cv2.resize(img, target_size)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch


def predict_freshness(image_path):
    """
    Run prediction on the uploaded image.
    Returns dict with prediction, freshness_score, confidence, food_name, category.
    """
    model = get_model()

    if model is None:
        # Demo mode: simulate a prediction for testing without trained model
        import random
        predictions_demo = [
            {'label': 'fresh_fruits', 'score': random.uniform(75, 95)},
            {'label': 'fresh_vegetables', 'score': random.uniform(65, 88)},
            {'label': 'rotten_fruits', 'score': random.uniform(20, 45)},
            {'label': 'rotten_vegetables', 'score': random.uniform(15, 40)},
        ]
        chosen = random.choice(predictions_demo[:2])  # Bias towards fresh for demo
        label = chosen['label']
        score = chosen['score']
    else:
        # Real model prediction
        img_batch = preprocess_image(image_path)
        raw_preds = model.predict(img_batch)[0]
        best_idx = int(np.argmax(raw_preds))
        label = CLASS_LABELS[best_idx]
        score = float(raw_preds[best_idx]) * 100

    # Map label to freshness
    if 'fresh' in label:
        prediction = 'Fresh'
        freshness_score = min(score, 98.0)
    elif 'rotten' in label:
        prediction = 'Rotten'
        freshness_score = max(100 - score, 5.0)
    else:
        prediction = 'Semi-Fresh'
        freshness_score = 55.0

    # Determine category
    category = 'fruit' if 'fruit' in label else 'vegetable'

    # Detect food name from filename or default
    filename = os.path.basename(str(image_path)).lower()
    food_name = 'Food Item'
    for key, name in FOOD_NAMES.items():
        if key in filename:
            food_name = name
            break
    if food_name == 'Food Item':
        food_name = 'Fresh Fruit' if 'fruit' in label else 'Vegetable'
        if 'rotten' in label:
            food_name = 'Rotten ' + food_name

    return {
        'prediction': prediction,
        'freshness_score': round(freshness_score, 1),
        'confidence': round(score, 1),
        'food_name': food_name,
        'category': category,
        'label': label,
    }


def calculate_shelf_life(prediction, freshness_score):
    """Calculate shelf life based on prediction and freshness score."""
    base = SHELF_LIFE_MAP[prediction]
    days = base['days']
    hours = base['hours']

    # Fine-tune based on freshness score
    if prediction == 'Fresh':
        if freshness_score > 90:
            days = 7
            hours = 0
        elif freshness_score > 80:
            days = 5
            hours = 12
        else:
            days = 3
            hours = 6
    elif prediction == 'Semi-Fresh':
        if freshness_score > 60:
            days = 2
            hours = 0
        else:
            days = 1
            hours = 0

    expiry_date = (datetime.now() + timedelta(days=days, hours=hours)).date()
    return days, hours, expiry_date


def get_storage_suggestion(prediction, category):
    """Get AI-based storage suggestion."""
    suggestions = FOOD_SUGGESTIONS.get(prediction, FOOD_SUGGESTIONS['Fresh'])
    return suggestions.get(category, suggestions['default'])


# ─────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────

def home(request):
    """Home page view."""
    total_scans = FoodScan.objects.count()
    fresh_count = FoodScan.objects.filter(prediction='Fresh').count()
    rotten_count = FoodScan.objects.filter(prediction='Rotten').count()
    recent_scans = FoodScan.objects.order_by('-timestamp')[:6]

    context = {
        'total_scans': total_scans,
        'fresh_count': fresh_count,
        'rotten_count': rotten_count,
        'recent_scans': recent_scans,
    }
    return render(request, 'home.html', context)


def upload(request):
    """Upload page - handles image upload and prediction."""
    form = FoodImageUploadForm()

    if request.method == 'POST':
        form = FoodImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']

            # Validate file size (max 10MB)
            if image_file.size > 10 * 1024 * 1024:
                form.add_error('image', 'File size must be under 10MB.')
                return render(request, 'upload.html', {'form': form})

            # Save image to media
            from django.core.files.storage import default_storage
            from django.core.files.base import ContentFile

            filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.name}"
            saved_path = default_storage.save(f'uploads/{filename}', ContentFile(image_file.read()))
            full_path = os.path.join(settings.MEDIA_ROOT, saved_path)

            # Run prediction
            result = predict_freshness(full_path)

            # Calculate shelf life
            days, hours, expiry_date = calculate_shelf_life(
                result['prediction'], result['freshness_score']
            )

            # Get storage suggestion
            suggestion = get_storage_suggestion(result['prediction'], result['category'])

            # Save to database
            scan = FoodScan.objects.create(
                image=saved_path,
                food_name=result['food_name'],
                food_category=result['category'],
                prediction=result['prediction'],
                freshness_score=result['freshness_score'],
                confidence_score=result['confidence'],
                shelf_life_days=days,
                shelf_life_hours=hours,
                expiry_date=expiry_date,
                storage_suggestion=suggestion,
                user_ip=request.META.get('REMOTE_ADDR'),
            )

            return redirect('result', scan_id=scan.id)

    return render(request, 'upload.html', {'form': form})


def result(request, scan_id):
    """Result page - displays prediction results."""
    scan = get_object_or_404(FoodScan, id=scan_id)

    # Determine color theme
    color_map = {
        'Fresh': {'bg': '#00C853', 'badge': 'success', 'text': 'Fresh & Safe to Eat'},
        'Semi-Fresh': {'bg': '#FFB300', 'badge': 'warning', 'text': 'Consume Soon'},
        'Rotten': {'bg': '#D32F2F', 'badge': 'danger', 'text': 'Not Safe for Consumption'},
    }
    colors_info = color_map.get(scan.prediction, color_map['Fresh'])

    context = {
        'scan': scan,
        'colors': colors_info,
    }
    return render(request, 'result.html', context)


def history(request):
    """Scan history page."""
    scans = FoodScan.objects.all().order_by('-timestamp')

    # Filter by prediction
    filter_by = request.GET.get('filter', 'all')
    if filter_by == 'fresh':
        scans = scans.filter(prediction='Fresh')
    elif filter_by == 'semi':
        scans = scans.filter(prediction='Semi-Fresh')
    elif filter_by == 'rotten':
        scans = scans.filter(prediction='Rotten')

    context = {
        'scans': scans,
        'filter_by': filter_by,
        'total': scans.count(),
    }
    return render(request, 'history.html', context)


def dashboard(request):
    """Smart analytics dashboard."""
    from django.db.models.functions import TruncDate
    from django.db.models import Count
    import json

    total = FoodScan.objects.count()
    fresh = FoodScan.objects.filter(prediction='Fresh').count()
    semi = FoodScan.objects.filter(prediction='Semi-Fresh').count()
    rotten = FoodScan.objects.filter(prediction='Rotten').count()
    avg_score = FoodScan.objects.aggregate(avg=Avg('freshness_score'))['avg'] or 0

    # Daily scans (last 7 days)
    daily_data = (
        FoodScan.objects
        .annotate(date=TruncDate('timestamp'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )[:14]

    daily_labels = [str(d['date']) for d in daily_data]
    daily_counts = [d['count'] for d in daily_data]

    # Category distribution
    category_data = (
        FoodScan.objects
        .values('food_category')
        .annotate(count=Count('id'))
    )

    cat_labels = [d['food_category'] for d in category_data]
    cat_counts = [d['count'] for d in category_data]

    # Recent scans
    recent = FoodScan.objects.order_by('-timestamp')[:10]

    context = {
        'total': total,
        'fresh': fresh,
        'semi': semi,
        'rotten': rotten,
        'avg_score': round(avg_score, 1),
        'fresh_pct': round((fresh / total * 100) if total else 0, 1),
        'rotten_pct': round((rotten / total * 100) if total else 0, 1),
        'daily_labels': json.dumps(daily_labels),
        'daily_counts': json.dumps(daily_counts),
        'cat_labels': json.dumps(cat_labels),
        'cat_counts': json.dumps(cat_counts),
        'pie_data': json.dumps([fresh, semi, rotten]),
        'recent': recent,
    }
    return render(request, 'dashboard.html', context)


def about(request):
    """About page."""
    return render(request, 'about.html')


def download_pdf(request, scan_id):
    """Generate and download PDF report for a scan."""
    scan = get_object_or_404(FoodScan, id=scan_id)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#00C853'),
        spaceAfter=12,
    )
    story.append(Paragraph("🌿 FreshScan AI Report", title_style))
    story.append(Spacer(1, 12))

    # Scan info table
    data = [
        ['Field', 'Value'],
        ['Food Name', scan.food_name],
        ['Prediction', scan.prediction],
        ['Freshness Score', f"{scan.freshness_score}%"],
        ['Confidence', f"{scan.confidence_score}%"],
        ['Shelf Life', f"{scan.shelf_life_days} days {scan.shelf_life_hours} hrs"],
        ['Expiry Date', scan.expiry_date.strftime('%d/%m/%Y') if scan.expiry_date else 'N/A'],
        ['Storage Tip', scan.storage_suggestion],
        ['Scan Date', scan.timestamp.strftime('%d/%m/%Y %H:%M')],
    ]

    table = Table(data, colWidths=[2 * inch, 4 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A1A2E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8F5E9')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#DEE2E6')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Footer
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)
    story.append(Paragraph("Generated by FreshScan AI – Food Freshness Detection System", footer_style))

    doc.build(story)
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="FreshScan_Report_{scan_id}.pdf"'
    return response


def generate_qr(request, scan_id):
    """Generate QR code for a scan report."""
    scan = get_object_or_404(FoodScan, id=scan_id)

    # QR content
    qr_data = (
        f"FreshScan AI Report\n"
        f"Food: {scan.food_name}\n"
        f"Status: {scan.prediction}\n"
        f"Score: {scan.freshness_score}%\n"
        f"Shelf Life: {scan.shelf_life_days}d {scan.shelf_life_hours}h\n"
        f"Expiry: {scan.expiry_date}\n"
        f"Tip: {scan.storage_suggestion}"
    )

    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color='#1A1A2E', back_color='white')

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='image/png')
    response['Content-Disposition'] = f'attachment; filename="FreshScan_QR_{scan_id}.png"'
    return response


@csrf_exempt
def webcam_predict(request):
    """
    Handle webcam frame prediction via AJAX.
    Receives base64 image, returns JSON prediction.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image', '')

            # Decode base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({'error': 'Invalid image'}, status=400)

            # Save temp file
            tmp_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'webcam_tmp.jpg')
            cv2.imwrite(tmp_path, img)

            # Predict
            result = predict_freshness(tmp_path)
            days, hours, expiry = calculate_shelf_life(result['prediction'], result['freshness_score'])
            suggestion = get_storage_suggestion(result['prediction'], result['category'])

            return JsonResponse({
                'prediction': result['prediction'],
                'freshness_score': result['freshness_score'],
                'confidence': result['confidence'],
                'food_name': result['food_name'],
                'shelf_life_days': days,
                'shelf_life_hours': hours,
                'expiry_date': str(expiry),
                'suggestion': suggestion,
                'success': True,
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'POST only'}, status=405)


def delete_scan(request, scan_id):
    """Delete a scan from history."""
    scan = get_object_or_404(FoodScan, id=scan_id)
    scan.delete()
    return redirect('history')