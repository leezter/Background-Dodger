// Import the background removal library from CDN
import { removeBackground as imglyRemoveBackground } from 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.1/+esm';

// =============================================================================
// Configuration
// =============================================================================

const FLUX_API_BASE = 'http://127.0.0.1:8000';

// Register Service Worker for offline support
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered:', registration.scope);
            })
            .catch((error) => {
                console.log('SW registration failed:', error);
            });
    });
}

// =============================================================================
// Background Remover - DOM Elements
// =============================================================================

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

// =============================================================================
// Navigation and Panel Switching
// =============================================================================

const navItems = document.querySelectorAll('.nav-item');
const panels = document.querySelectorAll('.panel');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetPanel = item.dataset.panel;

        // Update active state on nav items
        navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');

        // Switch panels
        panels.forEach(panel => {
            panel.classList.remove('active');
            if (panel.id === `${targetPanel}-panel`) {
                panel.classList.add('active');
            }
        });
    });
});

// =============================================================================
// Image Generator - DOM Elements
// =============================================================================

const modeButtons = document.querySelectorAll('.mode-btn');
const img2imgUpload = document.getElementById('img2img-upload');
const img2imgDropZone = document.getElementById('img2img-drop-zone');
const img2imgFileInput = document.getElementById('img2img-file-input');
const img2imgPreviewContainer = document.getElementById('img2img-preview-container');
const img2imgPreview = document.getElementById('img2img-preview');
const img2imgClear = document.getElementById('img2img-clear');
const strengthSlider = document.getElementById('strength-slider');
const strengthValue = document.getElementById('strength-value');
const modelSelect = document.getElementById('model-select');
const promptInput = document.getElementById('prompt-input');
const widthSelect = document.getElementById('width-select');
const heightSelect = document.getElementById('height-select');
const stepsInput = document.getElementById('steps-input');
const seedInput = document.getElementById('seed-input');
const guidanceSlider = document.getElementById('guidance-slider');
const guidanceValue = document.getElementById('guidance-value');
const generateBtn = document.getElementById('generate-btn');
const genStatus = document.getElementById('gen-status');
const genStatusText = document.getElementById('gen-status-text');
const genResult = document.getElementById('gen-result');
const genResultImage = document.getElementById('gen-result-image');
const genDownloadBtn = document.getElementById('gen-download-btn');
const genSeedDisplay = document.getElementById('gen-seed-display');

// Image Generator State
let generatorMode = 'text2img'; // 'text2img' or 'img2img'
let img2imgFile = null;
let generatedImageData = null;
let lastGeneratedSeed = null;

// =============================================================================
// Image Generator - Mode Toggle
// =============================================================================

modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        generatorMode = btn.dataset.mode;

        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        if (generatorMode === 'img2img') {
            img2imgUpload.classList.remove('hidden');
        } else {
            img2imgUpload.classList.add('hidden');
        }
    });
});

// =============================================================================
// Image Generator - Img2Img Upload
// =============================================================================

img2imgDropZone.addEventListener('click', () => img2imgFileInput.click());

img2imgFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImg2ImgFile(e.target.files[0]);
    }
});

img2imgDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    img2imgDropZone.classList.add('drag-over');
});

img2imgDropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    img2imgDropZone.classList.remove('drag-over');
});

img2imgDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    img2imgDropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0 && isValidImageType(files[0])) {
        handleImg2ImgFile(files[0]);
    }
});

img2imgClear.addEventListener('click', () => {
    img2imgFile = null;
    img2imgFileInput.value = '';
    img2imgPreviewContainer.classList.add('hidden');
    img2imgDropZone.classList.remove('hidden');
});

function handleImg2ImgFile(file) {
    img2imgFile = file;
    const url = URL.createObjectURL(file);
    img2imgPreview.src = url;
    img2imgDropZone.classList.add('hidden');
    img2imgPreviewContainer.classList.remove('hidden');
}

// =============================================================================
// Image Generator - Sliders
// =============================================================================

strengthSlider.addEventListener('input', () => {
    strengthValue.textContent = strengthSlider.value;
});

guidanceSlider.addEventListener('input', () => {
    guidanceValue.textContent = guidanceSlider.value;
});

// =============================================================================
// Image Generator - Generate Image
// =============================================================================

generateBtn.addEventListener('click', generateImage);

async function generateImage() {
    const prompt = promptInput.value.trim();

    if (!prompt) {
        showError('Please enter a prompt');
        return;
    }

    if (generatorMode === 'img2img' && !img2imgFile) {
        showError('Please upload a source image for Image to Image mode');
        return;
    }

    // Show loading state
    generateBtn.disabled = true;
    genStatus.classList.remove('hidden');
    genStatusText.textContent = 'Connecting to FLUX server...';

    try {
        // Check if server is available
        const healthCheck = await fetch(`${FLUX_API_BASE}/api/health`).catch(() => null);

        if (!healthCheck || !healthCheck.ok) {
            throw new Error('Cannot connect to FLUX server. Make sure the Python backend is running on port 8000.');
        }

        genStatusText.textContent = 'Generating image...';

        let response;

        if (generatorMode === 'text2img') {
            // Text to Image
            const requestBody = {
                prompt: prompt,
                width: parseInt(widthSelect.value),
                height: parseInt(heightSelect.value),
                guidance_scale: parseFloat(guidanceSlider.value),
            };

            if (stepsInput.value) {
                requestBody.num_inference_steps = parseInt(stepsInput.value);
            }

            if (seedInput.value) {
                requestBody.seed = parseInt(seedInput.value);
            }

            response = await fetch(`${FLUX_API_BASE}/api/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });
        } else {
            // Image to Image
            const formData = new FormData();
            formData.append('image', img2imgFile);
            formData.append('prompt', prompt);
            formData.append('strength', strengthSlider.value);
            formData.append('guidance_scale', guidanceSlider.value);

            if (stepsInput.value) {
                formData.append('num_inference_steps', stepsInput.value);
            }

            if (seedInput.value) {
                formData.append('seed', seedInput.value);
            }

            response = await fetch(`${FLUX_API_BASE}/api/img2img`, {
                method: 'POST',
                body: formData,
            });
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Generation failed');
        }

        // Display result
        generatedImageData = result.image;
        lastGeneratedSeed = result.seed;

        genResultImage.src = `data:image/png;base64,${result.image}`;
        genSeedDisplay.textContent = `Seed: ${result.seed}`;
        genResult.classList.remove('hidden');

    } catch (error) {
        console.error('Generation failed:', error);
        showError(error.message || 'Failed to generate image');
    } finally {
        generateBtn.disabled = false;
        genStatus.classList.add('hidden');
    }
}

// =============================================================================
// Image Generator - Download
// =============================================================================

genDownloadBtn.addEventListener('click', () => {
    if (!generatedImageData) return;

    // Convert base64 to blob
    const byteCharacters = atob(generatedImageData);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });

    // Download
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `flux_generation_${lastGeneratedSeed || Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// =============================================================================
// Image Generator - Fullscreen Toggle
// =============================================================================

let fullscreenOverlay = null;

genResultImage.addEventListener('dblclick', () => {
    if (!generatedImageData) return;
    toggleFullscreen();
});

function toggleFullscreen() {
    if (fullscreenOverlay) {
        // Exit fullscreen
        fullscreenOverlay.style.opacity = '0';
        setTimeout(() => {
            if (fullscreenOverlay) {
                document.body.removeChild(fullscreenOverlay);
                fullscreenOverlay = null;
            }
        }, 200);
        return;
    }

    // Create fullscreen overlay
    fullscreenOverlay = document.createElement('div');
    fullscreenOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.95);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        cursor: zoom-out;
        opacity: 0;
        transition: opacity 0.2s ease;
    `;

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${generatedImageData}`;
    img.style.cssText = `
        max-width: 95vw;
        max-height: 95vh;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    `;

    fullscreenOverlay.appendChild(img);
    document.body.appendChild(fullscreenOverlay);

    // Fade in
    requestAnimationFrame(() => {
        fullscreenOverlay.style.opacity = '1';
    });

    // Close on double-click or click
    fullscreenOverlay.addEventListener('dblclick', toggleFullscreen);
    fullscreenOverlay.addEventListener('click', toggleFullscreen);

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === 'Escape' && fullscreenOverlay) {
            toggleFullscreen();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
}

// =============================================================================
// Background Remover - Event Listeners
// =============================================================================

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

// =============================================================================
// Background Remover - Core Functions
// =============================================================================

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

// =============================================================================
// Shared - Error Toast
// =============================================================================

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
