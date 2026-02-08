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

const img2imgDropZone2 = document.getElementById('img2img-drop-zone-2');
const img2imgFileInput2 = document.getElementById('img2img-file-input-2');
const img2imgPreviewContainer2 = document.getElementById('img2img-preview-container-2');
const img2imgPreview2 = document.getElementById('img2img-preview-2');
const img2imgClear2 = document.getElementById('img2img-clear-2');
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
const upscaleDropZone = document.getElementById('upscale-drop-zone');
const upscaleFileInput = document.getElementById('upscale-file-input');
const upscalePreviewContainer = document.getElementById('upscale-preview-container');
const upscalePreview = document.getElementById('upscale-preview');
const upscaleClear = document.getElementById('upscale-clear');
const upscaleBtn = document.getElementById('upscale-btn');
const upscaleStatus = document.getElementById('upscale-status');
const upscaleStatusText = document.getElementById('upscale-status-text');
const upscaleResult = document.getElementById('upscale-result');
const upscaleResultImage = document.getElementById('upscale-result-image');
const upscaleDownloadBtn = document.getElementById('upscale-download-btn');
const upscaleFactor = document.getElementById('upscale-factor');
const upscaleFaceEnhance = document.getElementById('upscale-face-enhance');

let upscaleFile = null;
let upscaledImageData = null;

if (upscaleDropZone) {
    upscaleDropZone.addEventListener('click', () => upscaleFileInput.click());

    upscaleFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUpscaleFile(e.target.files[0]);
        }
    });

    upscaleDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        upscaleDropZone.classList.add('drag-over');
    });

    upscaleDropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        upscaleDropZone.classList.remove('drag-over');
    });

    upscaleDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        upscaleDropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0 && isValidImageType(files[0])) {
            handleUpscaleFile(files[0]);
        }
    });
}

if (upscaleClear) {
    upscaleClear.addEventListener('click', () => {
        upscaleFile = null;
        upscaleFileInput.value = '';
        upscalePreviewContainer.classList.add('hidden');
        upscaleDropZone.classList.remove('hidden');
    });
}

const genResultGallery = document.getElementById('gen-result-gallery');
const genDownloadAllBtn = document.getElementById('gen-download-all-btn');
const numImagesSlider = document.getElementById('num-images-slider');
const numImagesValue = document.getElementById('num-images-value');

// Image Generator State
let generatorMode = 'text2img'; // 'text2img' or 'img2img'
let img2imgFile = null;
let img2imgFile2 = null;
let generatedImages = []; // Array of {image: base64, seed: number}

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
// Image Generator - Img2Img Upload (Image 2)
// =============================================================================

img2imgDropZone2.addEventListener('click', () => img2imgFileInput2.click());

img2imgFileInput2.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImg2ImgFile2(e.target.files[0]);
    }
});

img2imgDropZone2.addEventListener('dragover', (e) => {
    e.preventDefault();
    img2imgDropZone2.classList.add('drag-over');
});

img2imgDropZone2.addEventListener('dragleave', (e) => {
    e.preventDefault();
    img2imgDropZone2.classList.remove('drag-over');
});

img2imgDropZone2.addEventListener('drop', (e) => {
    e.preventDefault();
    img2imgDropZone2.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0 && isValidImageType(files[0])) {
        handleImg2ImgFile2(files[0]);
    }
});

img2imgClear2.addEventListener('click', () => {
    img2imgFile2 = null;
    img2imgFileInput2.value = '';
    img2imgPreviewContainer2.classList.add('hidden');
    img2imgDropZone2.classList.remove('hidden');
});

function handleImg2ImgFile2(file) {
    img2imgFile2 = file;
    const url = URL.createObjectURL(file);
    img2imgPreview2.src = url;
    img2imgDropZone2.classList.add('hidden');
    img2imgPreviewContainer2.classList.remove('hidden');
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

numImagesSlider.addEventListener('input', () => {
    numImagesValue.textContent = numImagesSlider.value;
});

// =============================================================================
// Image Generator - Generate Image
// =============================================================================

generateBtn.addEventListener('click', generateImage);

async function generateImage() {
    const prompt = promptInput.value.trim();
    const numImages = parseInt(numImagesSlider.value);

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

        const imageText = numImages === 1 ? 'image' : `${numImages} images`;
        genStatusText.textContent = `Generating ${imageText}...`;

        let response;

        if (generatorMode === 'text2img') {
            // Text to Image
            const requestBody = {
                prompt: prompt,
                width: parseInt(widthSelect.value),
                height: parseInt(heightSelect.value),
                guidance_scale: parseFloat(guidanceSlider.value),
                num_images: numImages,
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
            if (img2imgFile2) {
                formData.append('image2', img2imgFile2);
            }
            formData.append('prompt', prompt);
            formData.append('strength', strengthSlider.value);
            formData.append('guidance_scale', guidanceSlider.value);
            formData.append('num_images', numImages);

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

        // Handle both single image and multiple images response
        if (result.images && Array.isArray(result.images)) {
            generatedImages = result.images;
        } else if (result.image) {
            // Backward compatibility: single image response
            generatedImages = [{ image: result.image, seed: result.seed }];
        }

        // Display results in gallery
        displayGallery(generatedImages);

        // Show result, hide placeholder
        const placeholder = document.getElementById('gen-result-placeholder');
        if (placeholder) placeholder.classList.add('hidden');
        genResult.classList.remove('hidden');

        // Show/hide download all button based on number of images
        if (generatedImages.length > 1) {
            genDownloadAllBtn.classList.remove('hidden');
        } else {
            genDownloadAllBtn.classList.add('hidden');
        }

    } catch (error) {
        console.error('Generation failed:', error);
        showError(error.message || 'Failed to generate image');
    } finally {
        generateBtn.disabled = false;
        genStatus.classList.add('hidden');
    }
}

// Display images in gallery grid
function displayGallery(images) {
    genResultGallery.innerHTML = '';

    // Set grid class based on number of images
    genResultGallery.className = 'gen-result-gallery';
    if (images.length === 1) {
        genResultGallery.classList.add('single-image');
    } else if (images.length === 2) {
        genResultGallery.classList.add('two-images');
    } else if (images.length <= 4) {
        genResultGallery.classList.add('multi-images');
    } else {
        genResultGallery.classList.add('many-images');
    }

    images.forEach((imgData, index) => {
        const item = document.createElement('div');
        item.className = 'gallery-item';
        item.innerHTML = `
            <div class="gallery-item-image-wrapper">
                <img src="data:image/png;base64,${imgData.image}" 
                     class="gallery-item-image" 
                     alt="Generated image ${index + 1}"
                     data-index="${index}">
            </div>
            <div class="gallery-item-footer">
                <span class="gallery-item-seed">Seed: ${imgData.seed}</span>
                <div class="gallery-item-actions">
                    <button class="gallery-item-btn gallery-item-transfer" data-index="${index}" title="Use this image as a source for Image-to-Image generation">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="17 1 21 5 17 9"></polyline>
                            <path d="M3 11V9a4 4 0 0 1 4-4h14"></path>
                            <polyline points="7 23 3 19 7 15"></polyline>
                            <path d="M21 13v2a4 4 0 0 1-4 4H3"></path>
                        </svg>
                        Use as Source
                    </button>
                    <button class="gallery-item-btn gallery-item-download" data-index="${index}" title="Download this image">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        Save
                    </button>
                </div>
            </div>
        `;
        genResultGallery.appendChild(item);

        // Add click handler for fullscreen view
        const img = item.querySelector('.gallery-item-image');
        img.addEventListener('dblclick', () => showFullscreen(`data:image/png;base64,${imgData.image}`));

        // Add download handler
        const downloadBtn = item.querySelector('.gallery-item-download');
        downloadBtn.addEventListener('click', () => downloadImage(index));

        // Add transfer handler
        const transferBtn = item.querySelector('.gallery-item-transfer');
        transferBtn.addEventListener('click', () => openSlotModal(index));
    });
}

// Download a single image by index
function downloadImage(index) {
    if (!generatedImages[index]) return;

    const imgData = generatedImages[index];
    const byteCharacters = atob(imgData.image);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });

    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `flux_generation_${imgData.seed}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
}

// Download all images
genDownloadAllBtn.addEventListener('click', () => {
    if (generatedImages.length === 0) return;

    generatedImages.forEach((_, index) => {
        setTimeout(() => downloadImage(index), index * 200); // Stagger downloads
    });
});

// =============================================================================
// Image Generator - Fullscreen Toggle
// =============================================================================

// =============================================================================
// Fullscreen Logic (Generic)
// =============================================================================

let fullscreenOverlay = null;

// Generic function to show any image in fullscreen
function showFullscreen(imageSrc) {
    if (fullscreenOverlay) {
        closeFullscreen();
        return;
    }

    if (!imageSrc) return;

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
    img.src = imageSrc;
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
    fullscreenOverlay.addEventListener('dblclick', closeFullscreen);
    fullscreenOverlay.addEventListener('click', closeFullscreen);

    // Close on Escape key
    const escHandler = (e) => {
        if (e.key === 'Escape') {
            closeFullscreen();
            document.removeEventListener('keydown', escHandler);
        }
    };
    document.addEventListener('keydown', escHandler);
}

function closeFullscreen() {
    if (fullscreenOverlay) {
        fullscreenOverlay.style.opacity = '0';
        setTimeout(() => {
            if (fullscreenOverlay) {
                document.body.removeChild(fullscreenOverlay);
                fullscreenOverlay = null;
            }
        }, 200);
    }
}

// =============================================================================
// Image Source Enlarge Listeners
// =============================================================================

// Img2Img Source 1
if (img2imgPreview) {
    img2imgPreview.addEventListener('dblclick', () => showFullscreen(img2imgPreview.src));
}
// Img2Img Source 2
if (img2imgPreview2) {
    img2imgPreview2.addEventListener('dblclick', () => showFullscreen(img2imgPreview2.src));
}
// Upscaler Source
if (upscalePreview) {
    upscalePreview.addEventListener('dblclick', () => showFullscreen(upscalePreview.src));
}
// Upscaler Result
if (upscaleResultImage) {
    upscaleResultImage.addEventListener('dblclick', () => showFullscreen(upscaleResultImage.src));
}


// =============================================================================
// Image Transfer (Slot Selection)
// =============================================================================

let slotModal = null;
let pendingTransferData = null; // { type: 'generated'|'upscaled', index: number, imageBase64: string, seed: number }

function createSlotModal() {
    if (slotModal) return;

    slotModal = document.createElement('div');
    slotModal.id = 'slot-selection-modal';
    slotModal.className = 'modal';
    slotModal.innerHTML = `
        <div class="modal-content">
            <h3 class="modal-title">Use as Source Image</h3>
            <p class="modal-text">Which slot would you like to transfer this image to?</p>
            <div class="modal-actions">
                <button class="btn btn-secondary" id="slot-modal-cancel">Cancel</button>
                <button class="btn btn-primary" id="slot-modal-1">Slot 1</button>
                <button class="btn btn-primary" id="slot-modal-2">Slot 2</button>
            </div>
        </div>
    `;
    document.body.appendChild(slotModal);

    const cancelBtn = slotModal.querySelector('#slot-modal-cancel');
    const slot1Btn = slotModal.querySelector('#slot-modal-1');
    const slot2Btn = slotModal.querySelector('#slot-modal-2');

    cancelBtn.addEventListener('click', closeSlotModal);
    slot1Btn.addEventListener('click', () => transferPendingImageToSlot(1));
    slot2Btn.addEventListener('click', () => transferPendingImageToSlot(2));

    slotModal.addEventListener('click', (e) => {
        if (e.target === slotModal) closeSlotModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && slotModal && slotModal.classList.contains('visible')) {
            closeSlotModal();
        }
    });
}

// Open modal for a generated image
function openSlotModal(index) {
    if (!slotModal) createSlotModal();
    if (!generatedImages[index]) return;

    pendingTransferData = {
        type: 'generated',
        data: generatedImages[index]
    };
    slotModal.classList.add('visible');
}

// Open modal for the upscaled image
const upscaleTransferBtn = document.getElementById('upscale-transfer-btn');
if (upscaleTransferBtn) {
    upscaleTransferBtn.addEventListener('click', () => {
        if (!upscaledImageData) return;

        if (!slotModal) createSlotModal();
        pendingTransferData = {
            type: 'upscaled',
            data: {
                image: upscaledImageData,
                seed: Date.now() // Mock seed for filename
            }
        };
        slotModal.classList.add('visible');
    });
}

function closeSlotModal() {
    if (slotModal) {
        slotModal.classList.remove('visible');
        pendingTransferData = null;
    }
}

function transferPendingImageToSlot(slotNumber) {
    if (!pendingTransferData || !pendingTransferData.data) return;

    const imgData = pendingTransferData.data;

    // Convert base64 to File object
    const byteCharacters = atob(imgData.image);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });

    const prefix = pendingTransferData.type === 'upscaled' ? 'upscaled' : 'generated';
    const file = new File([blob], `${prefix}_${imgData.seed}.png`, { type: 'image/png' });

    // Transfer to appropriate slot
    if (slotNumber === 1) {
        handleImg2ImgFile(file);
    } else {
        handleImg2ImgFile2(file);
    }

    // Switch to Img2Img mode
    generatorMode = 'img2img';

    // Update UI
    modeButtons.forEach(btn => {
        if (btn.dataset.mode === 'img2img') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    img2imgUpload.classList.remove('hidden');

    closeSlotModal();

    // Switch to Generator panel if not already there
    const generatorNav = document.querySelector('.nav-item[data-panel="image-gen"]');
    if (generatorNav) generatorNav.click();

    // Scroll to top of generator panel
    document.querySelector('.generator-card-content').scrollTop = 0;
}

// Initialize modal on load
createSlotModal();

// =============================================================================
// Video Generator - Configuration
// =============================================================================

const VIDEO_API_BASE = 'http://127.0.0.1:8001';

// =============================================================================
// Video Generator - DOM Elements
// =============================================================================

const videoSourceDropZone = document.getElementById('video-source-drop-zone');
const videoSourceFileInput = document.getElementById('video-source-file-input');
const videoSourcePreviewContainer = document.getElementById('video-source-preview-container');
const videoSourcePreview = document.getElementById('video-source-preview');
const videoSourceClear = document.getElementById('video-source-clear');
const videoPromptInput = document.getElementById('video-prompt-input');
const videoFramesSelect = document.getElementById('video-frames-select');
const videoFpsSelect = document.getElementById('video-fps-select');
const videoStepsInput = document.getElementById('video-steps-input');
const videoSeedInput = document.getElementById('video-seed-input');
const videoGuidanceSlider = document.getElementById('video-guidance-slider');
const videoGuidanceValue = document.getElementById('video-guidance-value');
const videoGenerateBtn = document.getElementById('video-generate-btn');
const videoGenStatus = document.getElementById('video-gen-status');
const videoGenStatusText = document.getElementById('video-gen-status-text');
const videoResult = document.getElementById('video-result');
const videoResultPlayer = document.getElementById('video-result-player');
const videoDownloadBtn = document.getElementById('video-download-btn');
const videoSeedDisplay = document.getElementById('video-seed-display');

// Video Generator State
let videoSourceFile = null;
let generatedVideoData = null;
let lastVideoSeed = null;

// =============================================================================
// Video Generator - Source Image Upload
// =============================================================================

videoSourceDropZone.addEventListener('click', () => videoSourceFileInput.click());

videoSourceFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleVideoSourceFile(e.target.files[0]);
    }
});

videoSourceDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    videoSourceDropZone.classList.add('drag-over');
});

videoSourceDropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    videoSourceDropZone.classList.remove('drag-over');
});

videoSourceDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    videoSourceDropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0 && isValidImageType(files[0])) {
        handleVideoSourceFile(files[0]);
    }
});

videoSourceClear.addEventListener('click', () => {
    videoSourceFile = null;
    videoSourceFileInput.value = '';
    videoSourcePreviewContainer.classList.add('hidden');
    videoSourceDropZone.classList.remove('hidden');
});

function handleVideoSourceFile(file) {
    videoSourceFile = file;
    const url = URL.createObjectURL(file);
    videoSourcePreview.src = url;
    videoSourceDropZone.classList.add('hidden');
    videoSourcePreviewContainer.classList.remove('hidden');
}

// =============================================================================
// Video Generator - Sliders
// =============================================================================

videoGuidanceSlider.addEventListener('input', () => {
    videoGuidanceValue.textContent = videoGuidanceSlider.value;
});

// =============================================================================
// Video Generator - Generate Video
// =============================================================================

videoGenerateBtn.addEventListener('click', generateVideo);

async function generateVideo() {
    const prompt = videoPromptInput.value.trim();

    if (!videoSourceFile) {
        showError('Please upload a source image first');
        return;
    }

    if (!prompt) {
        showError('Please enter a prompt describing how to animate the image');
        return;
    }

    // Show loading state
    videoGenerateBtn.disabled = true;
    videoGenStatus.classList.remove('hidden');
    videoGenStatusText.textContent = 'Connecting to CogVideoX server...';

    try {
        // Check if server is available
        const healthCheck = await fetch(`${VIDEO_API_BASE}/api/video/health`).catch(() => null);

        if (!healthCheck || !healthCheck.ok) {
            throw new Error('Cannot connect to video server. Make sure video_server.py is running on port 8001.');
        }

        const healthData = await healthCheck.json();
        if (!healthData.model_loaded) {
            videoGenStatusText.textContent = 'Loading CogVideoX model (this may take 1-2 minutes on first run)...';
        } else {
            videoGenStatusText.textContent = 'Generating video (this may take 2-5 minutes)...';
        }

        // Build form data with image for I2V
        const formData = new FormData();
        formData.append('image', videoSourceFile);
        formData.append('prompt', prompt);
        formData.append('num_frames', videoFramesSelect.value);
        formData.append('fps', videoFpsSelect.value);
        formData.append('num_inference_steps', videoStepsInput.value || '50');
        formData.append('guidance_scale', videoGuidanceSlider.value);

        if (videoSeedInput.value) {
            formData.append('seed', videoSeedInput.value);
        }

        const response = await fetch(`${VIDEO_API_BASE}/api/video/generate`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Video generation failed');
        }

        // Display result
        generatedVideoData = result.video;
        lastVideoSeed = result.seed;

        // Convert base64 to blob URL for video player
        const byteCharacters = atob(result.video);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'video/mp4' });
        const videoUrl = URL.createObjectURL(blob);

        videoResultPlayer.src = videoUrl;
        videoSeedDisplay.textContent = `Seed: ${result.seed}`;

        // Show result, hide placeholder
        const placeholder = document.getElementById('video-result-placeholder');
        if (placeholder) placeholder.classList.add('hidden');
        videoResult.classList.remove('hidden');

    } catch (error) {
        console.error('Video generation failed:', error);
        showError(error.message || 'Failed to generate video');
    } finally {
        videoGenerateBtn.disabled = false;
        videoGenStatus.classList.add('hidden');
    }
}

// =============================================================================
// Video Generator - Download
// =============================================================================

videoDownloadBtn.addEventListener('click', () => {
    if (!generatedVideoData) return;

    // Convert base64 to blob
    const byteCharacters = atob(generatedVideoData);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'video/mp4' });

    // Download
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `cogvideo_generation_${lastVideoSeed || Date.now()}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

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
downloadBtn.addEventListener('click', downloadBgRemovedImage);

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

function downloadBgRemovedImage() {
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
// =============================================================================
// Image Generator - Slot Selection Modal (Legacy - Removed)
// =============================================================================

// =============================================================================
// Upscaler - Logic (Restored)
// =============================================================================

if (upscaleBtn) {
    upscaleBtn.addEventListener('click', upscaleImage);
}

if (upscaleDownloadBtn) {
    upscaleDownloadBtn.addEventListener('click', () => {
        if (!upscaledImageData) return;

        const link = document.createElement('a');
        link.href = upscaleResultImage.src;
        link.download = `upscaled_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

function handleUpscaleFile(file) {
    upscaleFile = file;
    const url = URL.createObjectURL(file);
    upscalePreview.src = url;
    upscaleDropZone.classList.add('hidden');
    upscalePreviewContainer.classList.remove('hidden');
}

async function upscaleImage() {
    if (!upscaleFile) {
        showError('Please upload an image to upscale');
        return;
    }

    // Show loading state
    upscaleBtn.disabled = true;
    upscaleStatus.classList.remove('hidden');
    upscaleStatusText.textContent = 'Connecting to FLUX server...';

    try {
        // Check if server is available (Upscaler is on FLUX server port 8000)
        const healthCheck = await fetch(`${FLUX_API_BASE}/api/health`).catch(() => null);

        if (!healthCheck || !healthCheck.ok) {
            throw new Error('Cannot connect to FLUX server. Make sure the Python backend is running on port 8000.');
        }

        upscaleStatusText.textContent = 'Upscaling image (this may take 10-20 seconds)...';

        const formData = new FormData();
        formData.append('image', upscaleFile);
        formData.append('scale', upscaleFactor.value);
        formData.append('face_enhance', upscaleFaceEnhance.checked);

        const response = await fetch(`${FLUX_API_BASE}/api/upscale`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Upscaling failed');
        }

        // Display result
        upscaledImageData = result.image;
        upscaleResultImage.src = `data:image/png;base64,${result.image}`;

        // Show result, hide placeholder
        const placeholder = document.getElementById('upscale-result-placeholder');
        if (placeholder) placeholder.classList.add('hidden');
        upscaleResult.classList.remove('hidden');

    } catch (error) {
        console.error('Upscaling failed:', error);
        showError(error.message || 'Failed to upscale image');
    } finally {
        upscaleBtn.disabled = false;
        upscaleStatus.classList.add('hidden');
    }
}






