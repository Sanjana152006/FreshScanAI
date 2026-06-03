/**
 * static/js/main.js
 * FreshScan AI – Main JavaScript
 * Handles: image preview, drag-drop, webcam, dark mode, loader,
 *          score ring animation, charts, voice assistant, scroll animations
 */

'use strict';

/* ────────────────────────────────────────────
   1. DOM Ready
──────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initDarkMode();
  initUploadZone();
  initScoreRing();
  initScrollAnimations();
  initNavHighlight();
  initWebcam();
  initVoiceAssistant();
});


/* ────────────────────────────────────────────
   2. Dark Mode Toggle
──────────────────────────────────────────── */
function initDarkMode() {
  const btn  = document.getElementById('darkModeBtn');
  const body = document.body;

  // Restore preference
  if (localStorage.getItem('theme') === 'light') {
    body.classList.add('light-mode');
    if (btn) btn.textContent = '🌙';
  }

  if (!btn) return;
  btn.addEventListener('click', () => {
    body.classList.toggle('light-mode');
    const isLight = body.classList.contains('light-mode');
    btn.textContent = isLight ? '🌙' : '☀️';
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
  });
}


/* ────────────────────────────────────────────
   3. Image Upload Zone
──────────────────────────────────────────── */
function initUploadZone() {
  const zone    = document.getElementById('uploadZone');
  const input   = document.getElementById('foodImageInput');
  const preview = document.getElementById('imagePreviewContainer');
  const previewImg = document.getElementById('imagePreview');
  const form    = document.getElementById('uploadForm');
  const loader  = document.getElementById('scanLoader');

  if (!zone || !input) return;

  // Click on zone triggers file input
  zone.addEventListener('click', (e) => {
    if (e.target !== input) input.click();
  });

  // Drag & drop
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) {
      input.files = files;
      handleFileSelect(files[0]);
    }
  });

  // File input change
  input.addEventListener('change', () => {
    if (input.files.length) handleFileSelect(input.files[0]);
  });

  function handleFileSelect(file) {
    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      showAlert('Please upload a JPG or PNG image.', 'danger');
      return;
    }
    // Validate size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      showAlert('File size must be under 10MB.', 'danger');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      preview.style.display = 'block';
      zone.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  // Form submit → show loader
  if (form) {
    form.addEventListener('submit', () => {
      if (input.files.length && loader) {
        loader.classList.add('active');
      }
    });
  }
}


/* ────────────────────────────────────────────
   4. Freshness Score Ring Animation
──────────────────────────────────────────── */
function initScoreRing() {
  const ring = document.querySelector('.ring-fill');
  const numEl = document.querySelector('.ring-number');
  if (!ring) return;

  const score = parseFloat(ring.dataset.score || 0);
  const circumference = 440;  // 2 * π * 70 ≈ 440

  // Map score to offset
  const offset = circumference - (score / 100) * circumference;

  // Set colour based on score
  let strokeColor = '#00C853';
  if (score < 40) strokeColor = '#F44336';
  else if (score < 70) strokeColor = '#FFB300';

  ring.style.stroke = strokeColor;

  // Animate after short delay
  setTimeout(() => {
    ring.style.strokeDashoffset = offset;
  }, 200);

  // Count-up number animation
  if (numEl) {
    animateCounter(numEl, 0, score, 1200);
  }
}

function animateCounter(el, from, to, duration) {
  const start = performance.now();
  const isFloat = String(to).includes('.');

  function step(ts) {
    const elapsed = ts - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);  // ease-out cubic
    const current = from + (to - from) * eased;
    el.textContent = isFloat ? current.toFixed(1) : Math.round(current);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}


/* ────────────────────────────────────────────
   5. Scroll Reveal Animations
──────────────────────────────────────────── */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.glass-card, .fade-in, .slide-left, .slide-right').forEach(el => {
    // Only apply if not already animated by CSS
    if (!el.style.opacity) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(24px)';
      el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      observer.observe(el);
    }
  });
}


/* ────────────────────────────────────────────
   6. Active Nav Link Highlight
──────────────────────────────────────────── */
function initNavHighlight() {
  const links = document.querySelectorAll('.nav-link');
  const path  = window.location.pathname;

  links.forEach(link => {
    const href = link.getAttribute('href');
    if (href && path.startsWith(href) && href !== '/') {
      link.classList.add('active');
    } else if (href === '/' && path === '/') {
      link.classList.add('active');
    }
  });
}


/* ────────────────────────────────────────────
   7. Webcam Detection
──────────────────────────────────────────── */
let webcamStream = null;
let webcamInterval = null;

function initWebcam() {
  const startBtn  = document.getElementById('startWebcam');
  const stopBtn   = document.getElementById('stopWebcam');
  const captureBtn= document.getElementById('captureWebcam');
  const video     = document.getElementById('webcamVideo');
  const canvas    = document.getElementById('webcamCanvas');
  const resultDiv = document.getElementById('webcamResult');

  if (!startBtn || !video) return;

  startBtn.addEventListener('click', async () => {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      video.srcObject = webcamStream;
      video.play();
      startBtn.style.display  = 'none';
      stopBtn.style.display   = 'inline-flex';
      captureBtn.style.display= 'inline-flex';
      showAlert('Webcam started. Point at food and click Capture.', 'success');
    } catch (err) {
      showAlert('Cannot access webcam: ' + err.message, 'danger');
    }
  });

  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
      }
      video.srcObject = null;
      startBtn.style.display  = 'inline-flex';
      stopBtn.style.display   = 'none';
      captureBtn.style.display= 'none';
    });
  }

  if (captureBtn) {
    captureBtn.addEventListener('click', () => {
      if (!video.srcObject) return;

      const ctx = canvas.getContext('2d');
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg', 0.85);

      if (resultDiv) {
        resultDiv.innerHTML = '<div class="text-center py-3"><div class="loader-ring mx-auto"></div><p class="mt-2 text-secondary-c small">Analyzing...</p></div>';
      }

      // Send to Django view
      fetch('/webcam-predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData }),
      })
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          displayWebcamResult(data, resultDiv);
          speakResult(data.food_name, data.prediction);
        } else {
          if (resultDiv) resultDiv.innerHTML = `<p class="text-red">Error: ${data.error}</p>`;
        }
      })
      .catch(err => {
        if (resultDiv) resultDiv.innerHTML = `<p class="text-red">Network error: ${err.message}</p>`;
      });
    });
  }
}

function displayWebcamResult(data, container) {
  if (!container) return;

  const colorMap = { Fresh: 'fresh', 'Semi-Fresh': 'semi', Rotten: 'rotten' };
  const cls = colorMap[data.prediction] || 'fresh';
  const emojiMap = { Fresh: '✅', 'Semi-Fresh': '⚠️', Rotten: '❌' };

  container.innerHTML = `
    <div class="fade-in">
      <div class="result-badge ${cls} mb-3">
        ${emojiMap[data.prediction]} ${data.prediction}
      </div>
      <div class="row g-3 text-center">
        <div class="col-6">
          <p class="small text-muted-c mb-1">Freshness</p>
          <p class="fw-bold text-green">${data.freshness_score}%</p>
        </div>
        <div class="col-6">
          <p class="small text-muted-c mb-1">Confidence</p>
          <p class="fw-bold">${data.confidence}%</p>
        </div>
        <div class="col-6">
          <p class="small text-muted-c mb-1">Shelf Life</p>
          <p class="fw-bold">${data.shelf_life_days}d ${data.shelf_life_hours}h</p>
        </div>
        <div class="col-6">
          <p class="small text-muted-c mb-1">Expiry</p>
          <p class="fw-bold">${data.expiry_date}</p>
        </div>
      </div>
      <div class="divider"></div>
      <p class="small text-secondary-c">💡 ${data.suggestion}</p>
    </div>
  `;
}


/* ────────────────────────────────────────────
   8. Voice Assistant (Web Speech API)
──────────────────────────────────────────── */
function initVoiceAssistant() {
  const voiceBtn = document.getElementById('voiceReadBtn');
  if (!voiceBtn) return;

  voiceBtn.addEventListener('click', () => {
    const foodName   = voiceBtn.dataset.food   || 'food item';
    const prediction = voiceBtn.dataset.pred   || 'unknown';
    const score      = voiceBtn.dataset.score  || '0';
    const shelf      = voiceBtn.dataset.shelf  || '0 days';

    const text = `The uploaded ${foodName} appears ${prediction}. 
                  Freshness score is ${score} percent. 
                  Estimated shelf life remaining: ${shelf}.`;

    speakResult(foodName, prediction, text);
    voiceBtn.textContent = '🔊 Speaking...';
    setTimeout(() => { voiceBtn.textContent = '🎤 Read Aloud'; }, 4000);
  });
}

function speakResult(food, prediction, customText) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();

  const text = customText ||
    `The uploaded ${food} appears ${prediction}. Please act accordingly.`;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate   = 0.95;
  utterance.pitch  = 1.0;
  utterance.volume = 1.0;

  // Pick a pleasant voice if available
  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v => v.name.includes('Google') || v.lang === 'en-US');
  if (preferred) utterance.voice = preferred;

  window.speechSynthesis.speak(utterance);
}


/* ────────────────────────────────────────────
   9. Alert Helper
──────────────────────────────────────────── */
function showAlert(message, type = 'info') {
  const alertContainer = document.getElementById('alertContainer');
  if (!alertContainer) return;

  const colorMap = { success: 'var(--green)', danger: 'var(--red)', info: 'var(--blue-accent)', warning: 'var(--amber)' };

  const alert = document.createElement('div');
  alert.style.cssText = `
    background: var(--bg-card);
    border: 1px solid ${colorMap[type] || colorMap.info};
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 12px;
    color: var(--text-primary);
    font-size: 0.88rem;
    animation: fadeIn 0.3s ease;
    display: flex;
    align-items: center;
    gap: 10px;
  `;

  const icons = { success: '✅', danger: '❌', info: 'ℹ️', warning: '⚠️' };
  alert.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
  alertContainer.appendChild(alert);

  setTimeout(() => {
    alert.style.opacity = '0';
    alert.style.transition = 'opacity 0.3s';
    setTimeout(() => alert.remove(), 300);
  }, 4000);
}


/* ────────────────────────────────────────────
   10. Dashboard Charts (Chart.js)
──────────────────────────────────────────── */
function initDashboardCharts(pieData, dailyLabels, dailyCounts, catLabels, catCounts) {
  Chart.defaults.color = '#8A9BB5';
  Chart.defaults.font.family = "'DM Sans', sans-serif";

  // Pie chart – Fresh / Semi / Rotten
  const pieCtx = document.getElementById('freshnessDonut');
  if (pieCtx) {
    new Chart(pieCtx, {
      type: 'doughnut',
      data: {
        labels: ['Fresh', 'Semi-Fresh', 'Rotten'],
        datasets: [{
          data: pieData,
          backgroundColor: ['#00C853', '#FFB300', '#F44336'],
          borderColor: '#0A0F1E',
          borderWidth: 3,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        cutout: '72%',
        plugins: {
          legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} scans` } }
        }
      }
    });
  }

  // Line / Bar chart – Daily scans
  const lineCtx = document.getElementById('dailyScansChart');
  if (lineCtx) {
    new Chart(lineCtx, {
      type: 'bar',
      data: {
        labels: dailyLabels,
        datasets: [{
          label: 'Scans per Day',
          data: dailyCounts,
          backgroundColor: 'rgba(0,200,83,0.25)',
          borderColor: '#00C853',
          borderWidth: 2,
          borderRadius: 6,
          hoverBackgroundColor: 'rgba(0,200,83,0.45)',
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true, ticks: { stepSize: 1 } }
        }
      }
    });
  }

  // Category bar chart
  const catCtx = document.getElementById('categoryChart');
  if (catCtx) {
    new Chart(catCtx, {
      type: 'bar',
      data: {
        labels: catLabels,
        datasets: [{
          label: 'Scans',
          data: catCounts,
          backgroundColor: ['#00C853','#448AFF','#FFB300','#F44336','#AA00FF'],
          borderRadius: 8,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true },
          y: { grid: { display: false } }
        }
      }
    });
  }
}

// Expose globally so Django template can call it
window.initDashboardCharts = initDashboardCharts;