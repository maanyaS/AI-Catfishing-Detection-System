/**
 * CatfishGuard – Frontend Logic
 * Handles file upload, drag & drop, API calls, and results rendering
 */

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImage = document.getElementById('previewImage');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsArea = document.getElementById('resultsArea');

    let selectedFile = null;

    // ─── File Upload Handling ───

    uploadArea.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFile(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid image file (PNG, JPG, GIF, WEBP, or BMP).');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            alert('File is too large. Maximum size is 16MB.');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            uploadPreview.style.display = 'flex';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadContent.style.display = 'flex';
        uploadPreview.style.display = 'none';
        analyzeBtn.disabled = true;
        resultsArea.style.display = 'none';
    }

    // ─── Scan Again ───
    document.getElementById('scanAgainBtn').addEventListener('click', () => {
        resetUpload();
        document.getElementById('analyzer').scrollIntoView({ behavior: 'smooth' });
    });

    // ─── Analysis ───

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoading = analyzeBtn.querySelector('.btn-loading');

        // Show loading
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        analyzeBtn.disabled = true;
        resultsArea.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            renderResults(data);

        } catch (err) {
            console.error('Analysis error:', err);
            alert('Something went wrong. Please try again.');
        } finally {
            btnText.style.display = 'flex';
            btnLoading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    // ─── Render Results ───

    function renderResults(data) {
        resultsArea.style.display = 'block';

        // Animate gauge
        const gaugeArc = document.getElementById('gaugeArc');
        const gaugeValue = document.getElementById('gaugeValue');
        const totalLength = 251;
        const targetOffset = totalLength - (totalLength * (data.overall_score / 100));

        // Animate the score number
        animateNumber(gaugeValue, 0, data.overall_score, 1200);

        // Animate the arc
        setTimeout(() => {
            gaugeArc.style.transition = 'stroke-dashoffset 1.2s ease-out';
            gaugeArc.style.strokeDashoffset = targetOffset;
        }, 100);

        // Set gauge value color
        gaugeValue.style.color = data.risk.color;

        // Risk badge
        const riskBadge = document.getElementById('riskBadge');
        riskBadge.textContent = data.risk.level + ' RISK';
        riskBadge.style.background = data.risk.color + '20';
        riskBadge.style.color = data.risk.color;

        // Risk message
        document.getElementById('riskMessage').textContent = data.risk.message;

        // Analysis grid
        const analysisGrid = document.getElementById('analysisGrid');
        const analysisItems = [
            { label: 'Anatomy', key: 'anatomical_anomalies', color: '#818cf8' },
            { label: 'Background', key: 'background_coherence', color: '#f472b6' },
            { label: 'Skin', key: 'skin_perfection', color: '#34d399' },
            { label: 'Text/Objects', key: 'text_object_artifacts', color: '#fb923c' },
            { label: 'Lighting', key: 'lighting_consistency', color: '#fbbf24' },
            { label: 'Frequency', key: 'frequency_anomaly', color: '#a78bfa' },
            { label: 'Noise', key: 'noise_anomaly', color: '#22d3ee' }
        ];

        analysisGrid.innerHTML = analysisItems.map(item => {
            const val = data.analysis[item.key];
            const pct = Math.round(val * 100);
            const barColor = getBarColor(pct);

            return `
                <div class="analysis-card">
                    <div class="label">${item.label}</div>
                    <div class="value" style="color: ${barColor}">${pct}%</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${pct}%; background: ${barColor};"></div>
                    </div>
                </div>
            `;
        }).join('');

        // Warnings
        const warningsList = document.getElementById('warningsList');
        warningsList.innerHTML = data.warnings.map(w => `
            <div class="warning-item ${w.type}">
                <div class="warning-title">${w.icon} ${w.title}</div>
                <div class="warning-detail">${w.detail}</div>
            </div>
        `).join('');

        // Tips
        const tipsList = document.getElementById('tipsList');
        tipsList.innerHTML = data.tips.map(tip => `
            <div class="tip-item">${tip}</div>
        `).join('');

        // Scroll to results
        setTimeout(() => {
            resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);
    }

    function getBarColor(pct) {
        if (pct >= 70) return '#ef4444';
        if (pct >= 50) return '#f59e0b';
        if (pct >= 30) return '#3b82f6';
        return '#10b981';
    }

    function animateNumber(el, start, end, duration) {
        const startTime = performance.now();
        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * eased;
            el.textContent = Math.round(current * 10) / 10;
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }
});
