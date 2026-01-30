// Import the background removal library from CDN
import { removeBackground as imglyRemoveBackground } from 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.1/+esm';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultSection = document.getElementById('result-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const originalImage = document.getElementById('original-image');
const processedImage = document.getElementById('processed-image');
const originalDimensions = document.getElementById('original-dimensions');
const processedDimensions = document.getElementById('processed-dimensions');
const downloadBtn = document.getElementById('download-btn');
const newImageBtn = document.getElementById('new-image-btn');

// Store the processed blob for download
let processedBlob = null;
let originalFileName = 'image';

// ============================================
// Event Listeners
// ============================================

// Click to open file picker
dropZone.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidImageType(file)) {
            handleFile(file);
        } else {
            showError('Please drop a valid image file (PNG, JPG, JPEG, or WebP)');
        }
    }
});

// Download button
downloadBtn.addEventListener('click', downloadImage);

// New image button
newImageBtn.addEventListener('click', resetToUpload);

// ============================================
// Core Functions
// ============================================

function isValidImageType(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    return validTypes.includes(file.type);
}

function handleFile(file) {
    if (!isValidImageType(file)) {
        showError('Invalid file type. Please use PNG, JPG, JPEG, or WebP.');
        return;
    }

    // Store original filename (without extension)
    originalFileName = file.name.replace(/\.[^/.]+$/, '');

    // Show processing section
    showSection('processing');

    // Create URL for original image preview
    const originalUrl = URL.createObjectURL(file);
    originalImage.src = originalUrl;

    // Get original dimensions
    const img = new Image();
    img.onload = () => {
        originalDimensions.textContent = `${img.width} × ${img.height}px`;
    };
    img.src = originalUrl;

    // Process the image
    removeBackground(file);
}

async function removeBackground(file) {
    try {
        updateProgress(5, 'Loading AI model (first time may take 30-60 seconds)...');

        const config = {
            progress: (key, current, total) => {
                // Map progress to 5-90%
                const percent = 5 + Math.round((current / total) * 85);

                let statusText = 'Processing...';
                if (key.includes('fetch')) {
                    statusText = 'Downloading AI model...';
                } else if (key.includes('compute')) {
                    statusText = 'Analyzing image...';
                } else if (key.includes('inference')) {
                    statusText = 'Removing background...';
                }

                updateProgress(percent, statusText);
            }
        };

        // Process the image
        const blob = await imglyRemoveBackground(file, config);

        updateProgress(95, 'Finalizing...');

        // Store for download
        processedBlob = blob;

        // Display result
        const resultUrl = URL.createObjectURL(blob);
        processedImage.src = resultUrl;

        // Get processed dimensions
        const img = new Image();
        img.onload = () => {
            processedDimensions.textContent = `${img.width} × ${img.height}px`;
            updateProgress(100, 'Complete!');

            // Show result section after a brief delay
            setTimeout(() => showSection('result'), 300);
        };
        img.src = resultUrl;

    } catch (error) {
        console.error('Background removal failed:', error);
        showError('Failed to process image. Please try again.');
        resetToUpload();
    }
}

function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
}

function downloadImage() {
    if (!processedBlob) return;

    const link = document.createElement('a');
    link.href = URL.createObjectURL(processedBlob);
    link.download = `${originalFileName}_no_background.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showSection(section) {
    uploadSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultSection.classList.add('hidden');

    switch (section) {
        case 'upload':
            uploadSection.classList.remove('hidden');
            break;
        case 'processing':
            processingSection.classList.remove('hidden');
            progressFill.style.width = '0%';
            break;
        case 'result':
            resultSection.classList.remove('hidden');
            break;
    }
}

function resetToUpload() {
    // Clean up URLs
    if (originalImage.src) URL.revokeObjectURL(originalImage.src);
    if (processedImage.src) URL.revokeObjectURL(processedImage.src);

    // Reset state
    processedBlob = null;
    originalFileName = 'image';
    fileInput.value = '';
    originalImage.src = '';
    processedImage.src = '';
    originalDimensions.textContent = '';
    processedDimensions.textContent = '';

    showSection('upload');
}

function showError(message) {
    // Create temporary error toast
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-weight: 500;
        z-index: 1000;
        animation: slideUp 0.3s ease;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
    `;
    toast.textContent = message;

    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideUp {
            from { transform: translateX(-50%) translateY(20px); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideUp 0.3s ease reverse';
        setTimeout(() => {
            document.body.removeChild(toast);
            document.head.removeChild(style);
        }, 300);
    }, 3000);
}
