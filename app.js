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
const referenceSlotEls = Array.from(document.querySelectorAll('[data-ref-slot]'));
const strengthSlider = document.getElementById('strength-slider');
const strengthValue = document.getElementById('strength-value');
const modelSelect = document.getElementById('model-select');
const loraSelect = document.getElementById('lora-select');
const loraScaleSlider = document.getElementById('lora-scale-slider');
const loraScaleValue = document.getElementById('lora-scale-value');
const loraScaleContainer = document.getElementById('lora-scale-container');
const refreshLorasBtn = document.getElementById('refresh-loras-btn');
const promptInput = document.getElementById('prompt-input');
const promptComposer = document.getElementById('prompt-composer');
const promptModeSelect = document.getElementById('prompt-mode-select');
const promptRoleSummary = document.getElementById('prompt-role-summary');
const composedPromptInput = document.getElementById('composed-prompt-input');
const refreshComposedPromptBtn = document.getElementById('refresh-composed-prompt-btn');
const copyComposedPromptBtn = document.getElementById('copy-composed-prompt-btn');
const widthSelect = document.getElementById('width-select');
const heightSelect = document.getElementById('height-select');
const referenceResolutionSelect = document.getElementById('reference-resolution-select');
const outputResolutionSelect = document.getElementById('output-resolution-select');
const characterProfilePanel = document.getElementById('character-profile-panel');
const characterProfileSelect = document.getElementById('character-profile-select');
const characterProfileNameInput = document.getElementById('character-profile-name-input');
const saveCharacterProfileBtn = document.getElementById('save-character-profile-btn');
const loadCharacterProfileBtn = document.getElementById('load-character-profile-btn');
const deleteCharacterProfileBtn = document.getElementById('delete-character-profile-btn');
const characterIntentSelect = document.getElementById('character-intent-select');
const characterProfileStatus = document.getElementById('character-profile-status');
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
const upscaleDownload = document.getElementById('upscale-download');
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

if (upscaleDownload) {
    upscaleDownload.addEventListener('click', (e) => {
        e.stopPropagation();
        if (upscalePreview.src) downloadSourceImage(upscalePreview.src, 'upscale_source.png');
    });
}

const genResultGallery = document.getElementById('gen-result-gallery');
const genDownloadAllBtn = document.getElementById('gen-download-all-btn');
const numImagesSlider = document.getElementById('num-images-slider');
const numImagesValue = document.getElementById('num-images-value');

// Image Generator State
let generatorMode = 'text2img'; // 'text2img' or 'img2img'
const referenceSlots = referenceSlotEls.map((slotEl, index) => ({
    index,
    file: null,
    objectUrl: null,
    slotEl,
    roleSelect: document.getElementById(`reference-role-${index}`),
    dropZone: document.getElementById(`reference-drop-${index}`),
    fileInput: document.getElementById(`reference-file-${index}`),
    previewContainer: document.getElementById(`reference-preview-container-${index}`),
    previewImage: document.getElementById(`reference-preview-${index}`),
    clearBtn: document.getElementById(`reference-clear-${index}`),
    downloadBtn: document.getElementById(`reference-download-${index}`),
}));
let generatedImages = []; // Array of {image: base64, seed: number}
let selectedModelLoaded = false;

const modelCompatibilityNote = document.getElementById('model-compat-note');
let fluxModelApiInfo = {};
const CHARACTER_PROFILE_STORAGE_KEY = 'fluxCharacterProfiles';

const FLUX_MODEL_HELP = {
    'klein-4b': {
        label: 'FLUX.2 Klein 4B',
        note: 'Fast default for drafts and everyday images. Works with Text to Image, Image to Image, LoRA, seed, guidance, and up to 8 steps. Best guidance is usually 1.0.',
        defaultSteps: 4,
        defaultGuidance: 1.0,
        maxSteps: 8,
    },
    'klein-4b-fp8': {
        label: 'FLUX.2 Klein 4B FP8',
        note: 'FP8-derived Klein 4B transformer loaded through a Diffusers-compatible converted checkpoint. Works with the same controls as Klein 4B and is capped at 8 steps.',
        defaultSteps: 4,
        defaultGuidance: 1.0,
        maxSteps: 8,
    },
    'klein-9b': {
        label: 'FLUX.2 Klein 9B',
        note: 'Higher-quality Klein model for more detail and better prompt understanding. Requires Hugging Face gated-model access and backend authentication. Uses more VRAM and is capped at 8 steps.',
        defaultSteps: 4,
        defaultGuidance: 1.0,
        maxSteps: 8,
        warning: 'Requires Hugging Face access approval and a backend token.',
    },
    dev: {
        label: 'FLUX.2 Dev',
        note: 'Highest-quality option. Requires Hugging Face gated-model access, backend authentication, and H100-class GPU memory. Supports up to 50 steps. Guidance around 3.5 is usually a better starting point than 1.0.',
        defaultSteps: 28,
        defaultGuidance: 3.5,
        maxSteps: 50,
        warning: 'Requires Hugging Face access approval, a backend token, and H100-class GPU memory.',
    },
};

const REFERENCE_ROLE_TEXT = {
    primary: {
        label: 'Primary image',
        instruction: 'Use this reference as the main canvas, subject placement, composition anchor, and overall scene structure.',
        promptLine: 'Use reference image {n} as the primary canvas. Preserve its main composition, subject placement, perspective, and scene structure unless the prompt explicitly asks for a change.',
    },
    character: {
        label: 'Character identity',
        instruction: 'Preserve identity, facial structure, hair, body type, and recognizable character traits.',
        promptLine: 'Use reference image {n} only for character identity. Preserve the face, hair, body type, age, and recognizable traits from this reference.',
    },
    pose: {
        label: 'Pose / layout',
        instruction: 'Use pose, gesture, body position, camera angle, and layout without copying identity.',
        promptLine: 'Use reference image {n} only for pose, gesture, camera angle, framing, and body layout. Do not copy the person identity, clothing, face, or style from this pose reference.',
    },
    style: {
        label: 'Style reference',
        instruction: 'Use visual style, medium, lighting mood, color grading, and rendering treatment.',
        promptLine: 'Use reference image {n} only for visual style, lighting, color grading, medium, texture, and rendering treatment.',
    },
    product: {
        label: 'Product / object',
        instruction: 'Preserve the object shape, proportions, markings, materials, and important product details.',
        promptLine: 'Use reference image {n} for the product or object. Preserve its shape, proportions, markings, materials, and important design details.',
    },
    color: {
        label: 'Color palette',
        instruction: 'Use palette, color relationships, and dominant tones.',
        promptLine: 'Use reference image {n} only for color palette, dominant tones, and color relationships.',
    },
    texture: {
        label: 'Texture / material',
        instruction: 'Use surface finish, fabric, grain, texture, and material cues.',
        promptLine: 'Use reference image {n} only for surface texture, fabric, grain, material behavior, and finish.',
    },
    background: {
        label: 'Background / scene',
        instruction: 'Use environment, setting, depth, atmosphere, and scene context.',
        promptLine: 'Use reference image {n} only for background, environment, setting, depth, atmosphere, and scene context.',
    },
    reference: {
        label: 'Supporting reference',
        instruction: 'Use as supporting visual context.',
        promptLine: 'Use reference image {n} as supporting visual context.',
    },
};

const CHARACTER_INTENT_TEXT = {
    off: {
        label: 'Off',
        lines: [],
    },
    preserve_identity: {
        label: 'Preserve same character',
        lines: [
            'Maintain the same character identity across the final image.',
            'Preserve facial structure, hair, age, body type, and other recognizable identity markers from the character reference.',
            'Do not blend the character identity with pose, style, or background references.',
        ],
    },
    new_outfit: {
        label: 'Same character, new outfit',
        lines: [
            'Keep the same character identity, face, hair, age, and body type.',
            'Change only the outfit, styling, accessories, or wardrobe details requested by the prompt.',
            'Do not change the character into a different person.',
        ],
    },
    new_pose: {
        label: 'Same character, new pose',
        lines: [
            'Keep the same character identity while changing pose, gesture, framing, or camera angle.',
            'Use pose references only for body position and layout.',
            'Do not copy identity from pose references.',
        ],
    },
    new_scene: {
        label: 'Same character, new scene',
        lines: [
            'Keep the same character identity while placing the character into the requested scene.',
            'Adapt lighting and atmosphere to the new environment without changing the character.',
            'Preserve recognizable face, hair, proportions, and character-defining details.',
        ],
    },
    character_sheet: {
        label: 'Character sheet',
        lines: [
            'Create a clean character sheet for the same character.',
            'Show consistent identity across multiple views or expressions in one coherent sheet.',
            'Use neutral lighting and clear full-body or portrait views unless the prompt says otherwise.',
        ],
    },
};

// =============================================================================
// Progress Polling Utility
// =============================================================================

let progressInterval = null;

function startProgressPolling(headerRealId, headerPlaceholderId) {
    const realEl = document.getElementById(headerRealId);
    const placeholderEl = document.getElementById(headerPlaceholderId);

    const updateTitleText = (text) => {
        document.querySelectorAll('.title-text').forEach(s => s.textContent = text);
    };

    // Initial state before polling loop kicks in
    updateTitleText('Igniting AI Engine...');

    // Reset and activate
    if (realEl) {
        realEl.style.width = '0%';
        realEl.classList.add('active');
    }
    if (placeholderEl) {
        placeholderEl.style.width = '0%';
        placeholderEl.classList.add('active');
    }

    if (progressInterval) clearInterval(progressInterval);

    progressInterval = setInterval(async () => {
        try {
            const res = await fetch(`${FLUX_API_BASE}/api/progress`);
            if (res.ok) {
                const data = await res.json();
                if (data.is_running && data.total_steps > 0) {
                    const percent = Math.min(100, (data.step / data.total_steps) * 100);

                    if (data.step === 0) {
                        // Still initializing
                        updateTitleText('Igniting AI Engine...');

                        if (realEl) {
                            realEl.classList.add('initializing');
                            realEl.classList.remove('active');
                        }
                        if (placeholderEl) {
                            placeholderEl.classList.add('initializing');
                            placeholderEl.classList.remove('active');
                        }
                    } else {
                        // Progress started
                        updateTitleText('Generating Masterpiece... ⚡');

                        if (realEl) {
                            realEl.classList.remove('initializing');
                            realEl.classList.add('active');
                            realEl.style.width = `${percent}%`;
                        }
                        if (placeholderEl) {
                            placeholderEl.classList.remove('initializing');
                            placeholderEl.classList.add('active');
                            placeholderEl.style.width = `${percent}%`;
                        }
                    }
                }
            }
        } catch (e) {
            // Ignore polling errors
        }
    }, 500);
}

function stopProgressPolling() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

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

        updateImageGenerationCompatibility();
        updatePromptComposer({ force: false });
    });
});

function updateImageGenerationCompatibility() {
    const selectedModel = FLUX_MODEL_HELP[modelSelect.value];
    const modeText = generatorMode === 'img2img'
        ? 'Image to Image is active. Width and Height are ignored in this mode; the uploaded source image decides the working size. Strength is currently not applied by the backend.'
        : 'Text to Image is active. Width and Height control the generated image size.';

    if (modelCompatibilityNote && selectedModel) {
        const warningText = selectedModel.warning ? ` ${selectedModel.warning}` : '';
        const apiModelInfo = fluxModelApiInfo[modelSelect.value];
        const accessText = apiModelInfo?.access_note ? ` ${apiModelInfo.access_note}` : '';
        modelCompatibilityNote.textContent = `${selectedModel.note} ${modeText}${warningText}${accessText}`;
        modelCompatibilityNote.classList.toggle('warning', Boolean(selectedModel.warning || generatorMode === 'img2img'));
    }

    if (selectedModel) {
        stepsInput.max = String(selectedModel.maxSteps);
        stepsInput.placeholder = `Auto (${selectedModel.defaultSteps})`;
        stepsInput.title = `This model accepts up to ${selectedModel.maxSteps} steps. Leave blank for its default.`;
        if (stepsInput.value && Number(stepsInput.value) > selectedModel.maxSteps) {
            stepsInput.value = String(selectedModel.maxSteps);
        }
        guidanceSlider.title = `Suggested starting guidance for ${selectedModel.label}: ${selectedModel.defaultGuidance}. Klein models usually prefer low guidance; Dev usually benefits from higher guidance.`;
    }

    const img2imgActive = generatorMode === 'img2img';
    [widthSelect, heightSelect].forEach(control => {
        control.disabled = img2imgActive;
        control.title = img2imgActive
            ? 'Disabled in Image to Image mode because the backend uses the uploaded source image size instead.'
            : 'Used in Text to Image mode to set the generated image size.';
    });
}

async function syncCurrentFluxModel() {
    try {
        const response = await fetch(`${FLUX_API_BASE}/api/models`);
        if (!response.ok) return;

        const data = await response.json();
        fluxModelApiInfo = Object.fromEntries((data.models || []).map(model => [model.key, model]));
        if (data.current && FLUX_MODEL_HELP[data.current]) {
            modelSelect.value = data.current;
            selectedModelLoaded = true;
        } else {
            selectedModelLoaded = false;
        }
    } catch (error) {
        selectedModelLoaded = false;
        console.warn('Could not read current FLUX model:', error);
    } finally {
        updateImageGenerationCompatibility();
    }
}

async function switchFluxModel() {
    const selectedModel = modelSelect.value;
    const modelInfo = FLUX_MODEL_HELP[selectedModel];
    const apiModelInfo = fluxModelApiInfo[selectedModel];
    generateBtn.disabled = true;
    genStatus.classList.remove('hidden');
    const accessText = apiModelInfo?.requires_auth ? ' This model is gated and requires Hugging Face authentication.' : '';
    genStatusText.textContent = `Loading ${modelInfo?.label || 'selected FLUX.2 model'}...${accessText}`;

    try {
        const response = await fetch(`${FLUX_API_BASE}/api/load-model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model: selectedModel }),
        });

        const result = await response.json().catch(() => ({}));
        if (!response.ok || result.success === false) {
            throw new Error(result.detail || result.error || 'Model loading failed');
        }

        selectedModelLoaded = true;
        genStatusText.textContent = `${modelInfo?.label || 'Selected model'} loaded.`;
        setTimeout(() => genStatus.classList.add('hidden'), 1600);
    } catch (error) {
        selectedModelLoaded = false;
        genStatusText.textContent = `Could not load ${modelInfo?.label || 'model'}: ${error.message}`;
    } finally {
        generateBtn.disabled = !selectedModelLoaded;
        updateImageGenerationCompatibility();
    }
}

modelSelect.addEventListener('change', () => {
    selectedModelLoaded = false;
    updateImageGenerationCompatibility();
    switchFluxModel();
});

syncCurrentFluxModel();

// =============================================================================
// Image Generator - Multi-reference Upload
// =============================================================================

function getActiveReferenceSlots() {
    return referenceSlots.filter(slot => slot.file);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
    });
}

function imageElementFromDataUrl(dataUrl) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = dataUrl;
    });
}

async function fileToProfileDataUrl(file, maxSide = 768) {
    const sourceDataUrl = await readFileAsDataUrl(file);
    const image = await imageElementFromDataUrl(sourceDataUrl);
    const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0, width, height);
    return canvas.toDataURL('image/jpeg', 0.9);
}

function dataUrlToFile(dataUrl, filename, fallbackType = 'image/png') {
    const [header, base64Data] = dataUrl.split(',');
    const mimeMatch = header.match(/data:(.*?);base64/);
    const mimeType = mimeMatch?.[1] || fallbackType;
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    return new File([new Uint8Array(byteNumbers)], filename, { type: mimeType });
}

function getCharacterProfiles() {
    try {
        const parsed = JSON.parse(localStorage.getItem(CHARACTER_PROFILE_STORAGE_KEY) || '[]');
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.warn('Could not read character profiles:', error);
        return [];
    }
}

function setCharacterProfiles(profiles) {
    localStorage.setItem(CHARACTER_PROFILE_STORAGE_KEY, JSON.stringify(profiles));
}

function updateCharacterProfileStatus(message) {
    if (characterProfileStatus) {
        characterProfileStatus.textContent = message || 'Profiles are saved locally in this browser.';
    }
}

function renderCharacterProfiles(selectedId = characterProfileSelect?.value || '') {
    if (!characterProfileSelect) return;

    const profiles = getCharacterProfiles().sort((a, b) => a.name.localeCompare(b.name));
    characterProfileSelect.innerHTML = '';
    if (profiles.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No saved profiles';
        characterProfileSelect.appendChild(option);
        return;
    }

    profiles.forEach(profile => {
        const option = document.createElement('option');
        option.value = profile.id;
        option.textContent = `${profile.name} (${profile.references.length})`;
        characterProfileSelect.appendChild(option);
    });

    if (selectedId && profiles.some(profile => profile.id === selectedId)) {
        characterProfileSelect.value = selectedId;
    }
}

async function saveCharacterProfile() {
    const activeSlots = getActiveReferenceSlots();
    if (activeSlots.length === 0) {
        updateCharacterProfileStatus('Add at least one reference image before saving a profile.');
        return;
    }

    const name = characterProfileNameInput?.value.trim() || `Character ${new Date().toLocaleString()}`;
    const references = [];
    for (const slot of activeSlots) {
        references.push({
            slotIndex: slot.index,
            role: slot.roleSelect?.value || 'character',
            fileName: slot.file.name.replace(/\.[^.]+$/, '.jpg'),
            type: 'image/jpeg',
            dataUrl: await fileToProfileDataUrl(slot.file),
        });
    }

    const profiles = getCharacterProfiles();
    const existingIndex = profiles.findIndex(profile => profile.name.toLowerCase() === name.toLowerCase());
    const profile = {
        id: existingIndex >= 0 ? profiles[existingIndex].id : `profile_${Date.now()}`,
        name,
        references,
        createdAt: existingIndex >= 0 ? profiles[existingIndex].createdAt : new Date().toISOString(),
        updatedAt: new Date().toISOString(),
    };

    if (existingIndex >= 0) {
        profiles[existingIndex] = profile;
    } else {
        profiles.push(profile);
    }

    setCharacterProfiles(profiles);
    renderCharacterProfiles(profile.id);
    updateCharacterProfileStatus(`Saved "${name}" locally with ${references.length} reference image${references.length === 1 ? '' : 's'}.`);
}

function loadCharacterProfile(profileId = characterProfileSelect?.value) {
    const profile = getCharacterProfiles().find(item => item.id === profileId);
    if (!profile) {
        updateCharacterProfileStatus('Choose a saved profile to load.');
        return;
    }

    referenceSlots.forEach(slot => clearReferenceSlot(slot.index, { silent: true }));
    profile.references.slice(0, referenceSlots.length).forEach((reference, index) => {
        const slotIndex = Math.min(index, referenceSlots.length - 1);
        const slot = referenceSlots[slotIndex];
        if (slot?.roleSelect) {
            slot.roleSelect.value = reference.role || (slotIndex === 0 ? 'character' : 'reference');
        }
        setReferenceFile(
            slotIndex,
            dataUrlToFile(reference.dataUrl, reference.fileName || `profile_reference_${slotIndex + 1}.png`, reference.type),
            { silent: true },
        );
    });

    if (characterProfileNameInput) {
        characterProfileNameInput.value = profile.name;
    }
    if (characterIntentSelect && characterIntentSelect.value === 'off') {
        characterIntentSelect.value = 'preserve_identity';
    }

    updatePromptComposer({ force: true });
    updateCharacterProfileStatus(`Loaded "${profile.name}" into the reference slots.`);
}

function deleteCharacterProfile() {
    const profileId = characterProfileSelect?.value;
    const profiles = getCharacterProfiles();
    const profile = profiles.find(item => item.id === profileId);
    if (!profile) {
        updateCharacterProfileStatus('Choose a saved profile to delete.');
        return;
    }

    setCharacterProfiles(profiles.filter(item => item.id !== profileId));
    renderCharacterProfiles();
    updateCharacterProfileStatus(`Deleted "${profile.name}" from local profiles.`);
}

function buildReferencePromptData() {
    const activeSlots = getActiveReferenceSlots();
    return activeSlots.map((slot, index) => {
        const role = slot.roleSelect?.value || 'reference';
        const roleText = REFERENCE_ROLE_TEXT[role] || REFERENCE_ROLE_TEXT.reference;
        return {
            index,
            slotNumber: slot.index + 1,
            imageNumber: index + 1,
            role,
            label: roleText.label,
            instruction: roleText.instruction,
            promptLine: roleText.promptLine.replace('{n}', String(index + 1)),
            fileName: slot.file?.name || `Reference ${slot.index + 1}`,
        };
    });
}

function composePrompt({ mode = 'composed' } = {}) {
    const basePrompt = promptInput.value.trim();
    const references = buildReferencePromptData();
    const intent = CHARACTER_INTENT_TEXT[characterIntentSelect?.value || 'off'] || CHARACTER_INTENT_TEXT.off;

    if (mode === 'original' || generatorMode !== 'img2img' || references.length === 0) {
        return basePrompt;
    }

    const roleLines = references.map(ref => ref.promptLine);
    if (mode === 'roles') {
        return [
            basePrompt,
            '',
            'Reference control instructions:',
            ...roleLines,
            ...(intent.lines.length ? ['', `Character consistency intent: ${intent.label}`, ...intent.lines] : []),
            'Do not swap identity, pose, style, product, color, texture, or background roles between references.',
        ].filter(Boolean).join('\n');
    }

    const hasCharacter = references.some(ref => ref.role === 'character');
    const hasPose = references.some(ref => ref.role === 'pose');
    const hasPrimary = references.some(ref => ref.role === 'primary');
    const guardrails = [
        'Create one coherent, polished image that follows the user prompt and uses each reference only for its assigned role.',
        hasPrimary ? 'Keep the primary reference as the strongest composition anchor.' : 'Use the first active reference as the main visual anchor when no primary reference is selected.',
        hasCharacter ? 'Keep character identity consistent and avoid blending faces from other references.' : '',
        hasPose ? 'Use pose references for body position and framing only; do not copy their identity unless the same reference is also labeled as character.' : '',
        'Resolve conflicts in favor of the written prompt, then the primary image, then the role-specific references.',
        'Avoid distorted anatomy, duplicated faces, mixed identities, unreadable text, and inconsistent lighting.',
    ].filter(Boolean);

    return [
        basePrompt || 'Create a high-quality image using the provided references.',
        '',
        'Reference plan:',
        ...roleLines,
        ...(intent.lines.length ? ['', `Character consistency intent: ${intent.label}`, ...intent.lines] : []),
        '',
        'Composition and quality instructions:',
        ...guardrails,
    ].join('\n');
}

function updatePromptComposer({ force = false } = {}) {
    if (!promptComposer) return;

    const references = buildReferencePromptData();
    const shouldShow = generatorMode === 'img2img';
    promptComposer.classList.toggle('hidden', !shouldShow);
    if (!shouldShow) return;

    if (promptRoleSummary) {
        if (references.length === 0) {
            promptRoleSummary.innerHTML = '<span>No active reference images yet.</span>';
        } else {
            const intent = CHARACTER_INTENT_TEXT[characterIntentSelect?.value || 'off'] || CHARACTER_INTENT_TEXT.off;
            const intentPill = intent.lines.length
                ? `<span class="prompt-role-pill character-intent-pill" title="${escapeHtml(intent.lines.join(' '))}">${escapeHtml(intent.label)}</span>`
                : '';
            promptRoleSummary.innerHTML = references.map(ref => `
                <span class="prompt-role-pill" title="${escapeHtml(ref.instruction)}">
                    ${ref.imageNumber}. ${escapeHtml(ref.label)}
                </span>
            `).join('') + intentPill;
        }
    }

    if (composedPromptInput && (force || !composedPromptInput.dataset.userEdited)) {
        composedPromptInput.value = composePrompt({ mode: 'composed' });
        composedPromptInput.dataset.userEdited = '';
    }
}

function getPromptForGeneration() {
    if (generatorMode !== 'img2img') return promptInput.value.trim();
    const mode = promptModeSelect?.value || 'composed';
    if (mode === 'composed') {
        return (composedPromptInput?.value || composePrompt({ mode: 'composed' })).trim();
    }
    return composePrompt({ mode }).trim();
}

function setReferenceFile(slotIndex, file, { silent = false } = {}) {
    const slot = referenceSlots[slotIndex];
    if (!slot || !file || !isValidImageType(file)) return;

    if (slot.objectUrl) {
        URL.revokeObjectURL(slot.objectUrl);
    }

    slot.file = file;
    slot.objectUrl = URL.createObjectURL(file);
    slot.previewImage.src = slot.objectUrl;
    slot.slotEl.classList.add('has-image');
    slot.dropZone.classList.add('hidden');
    slot.previewContainer.classList.remove('hidden');
    if (!silent) updatePromptComposer({ force: false });
}

function clearReferenceSlot(slotIndex, { silent = false } = {}) {
    const slot = referenceSlots[slotIndex];
    if (!slot) return;

    if (slot.objectUrl) {
        URL.revokeObjectURL(slot.objectUrl);
    }

    slot.file = null;
    slot.objectUrl = null;
    slot.fileInput.value = '';
    slot.previewImage.removeAttribute('src');
    slot.slotEl.classList.remove('has-image');
    slot.previewContainer.classList.add('hidden');
    slot.dropZone.classList.remove('hidden');
    if (!silent) updatePromptComposer({ force: false });
}

function setupReferenceSlot(slot) {
    if (!slot.dropZone || !slot.fileInput) return;

    slot.dropZone.addEventListener('click', () => slot.fileInput.click());

    slot.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            setReferenceFile(slot.index, e.target.files[0]);
        }
    });

    slot.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        slot.dropZone.classList.add('drag-over');
    });

    slot.dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        slot.dropZone.classList.remove('drag-over');
    });

    slot.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        slot.dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            setReferenceFile(slot.index, files[0]);
        }
    });

    slot.clearBtn?.addEventListener('click', () => clearReferenceSlot(slot.index));

    slot.downloadBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (slot.previewImage.src) {
            downloadSourceImage(slot.previewImage.src, `reference_${slot.index + 1}.png`);
        }
    });

    slot.previewImage?.addEventListener('dblclick', () => showFullscreen(slot.previewImage.src));
    slot.roleSelect?.addEventListener('change', () => updatePromptComposer({ force: false }));
}

referenceSlots.forEach(setupReferenceSlot);

promptInput.addEventListener('input', () => updatePromptComposer({ force: false }));
promptModeSelect?.addEventListener('change', () => updatePromptComposer({ force: false }));
composedPromptInput?.addEventListener('input', () => {
    composedPromptInput.dataset.userEdited = 'true';
});
refreshComposedPromptBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    if (composedPromptInput) {
        composedPromptInput.dataset.userEdited = '';
    }
    updatePromptComposer({ force: true });
});
copyComposedPromptBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    if (composedPromptInput?.value) {
        promptInput.value = composedPromptInput.value;
        composedPromptInput.dataset.userEdited = '';
        updatePromptComposer({ force: true });
    }
});
saveCharacterProfileBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    saveCharacterProfile().catch(error => {
        console.error('Could not save character profile:', error);
        updateCharacterProfileStatus('Could not save profile. The browser may be out of local storage space.');
    });
});
loadCharacterProfileBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    loadCharacterProfile();
});
deleteCharacterProfileBtn?.addEventListener('click', (e) => {
    e.preventDefault();
    deleteCharacterProfile();
});
characterIntentSelect?.addEventListener('change', () => updatePromptComposer({ force: false }));
characterProfileSelect?.addEventListener('change', () => {
    const profile = getCharacterProfiles().find(item => item.id === characterProfileSelect.value);
    updateCharacterProfileStatus(profile
        ? `"${profile.name}" has ${profile.references.length} saved reference image${profile.references.length === 1 ? '' : 's'}.`
        : 'Profiles are saved locally in this browser.');
});

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
// LoRA Logic
// =============================================================================

async function fetchLoras() {
    try {
        const response = await fetch(`${FLUX_API_BASE}/api/loras`);
        if (response.ok) {
            const data = await response.json();
            const currentLora = data.current_lora;

            // Clear existing options (keep None)
            loraSelect.innerHTML = '<option value="">None</option>';

            data.loras.forEach(lora => {
                const option = document.createElement('option');
                option.value = lora;
                option.textContent = lora.replace('.safetensors', '');
                if (lora === currentLora) option.selected = true;
                loraSelect.appendChild(option);
            });

            // Show slider if LoRA selected
            if (currentLora) {
                loraScaleContainer.classList.remove('hidden');
            }
        }
    } catch (error) {
        console.error('Failed to fetch LoRAs:', error);
    }
}

// Initial fetch
fetchLoras();

refreshLorasBtn.addEventListener('click', (e) => {
    e.preventDefault(); // Prevent accidental form submit if inside form
    fetchLoras();
    // Add simple animation
    refreshLorasBtn.classList.add('rotating');
    setTimeout(() => refreshLorasBtn.classList.remove('rotating'), 500);
});

loraSelect.addEventListener('change', () => {
    if (loraSelect.value) {
        loraScaleContainer.classList.remove('hidden');
    } else {
        loraScaleContainer.classList.add('hidden');
    }
});

loraScaleSlider.addEventListener('input', () => {
    loraScaleValue.textContent = loraScaleSlider.value;
});

// =============================================================================
// Image Generator - Generate Image
// =============================================================================

generateBtn.addEventListener('click', generateImage);

promptInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        generateImage();
    }
});

async function generateImage() {
    const prompt = getPromptForGeneration();
    const numImages = parseInt(numImagesSlider.value);

    if (!prompt) {
        showError('Please enter a prompt');
        return;
    }

    const activeReferenceSlots = getActiveReferenceSlots();

    if (generatorMode === 'img2img' && activeReferenceSlots.length === 0) {
        showError('Please add at least one reference image for Image to Image mode');
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

        startProgressPolling('gen-header-progress', 'gen-header-progress-placeholder');

        let response;

        if (generatorMode === 'text2img') {
            // Text to Image
            const requestBody = {
                model: modelSelect.value,
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

            // Add LoRA params
            if (loraSelect.value) {
                requestBody.lora_name = loraSelect.value;
                requestBody.lora_scale = parseFloat(loraScaleSlider.value);
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
            activeReferenceSlots.forEach(slot => {
                formData.append('reference_images', slot.file);
                formData.append('reference_roles', slot.roleSelect?.value || 'reference');
            });
            formData.append('model', modelSelect.value);
            formData.append('prompt', prompt);
            formData.append('prompt_mode', promptModeSelect?.value || 'composed');
            formData.append('strength', strengthSlider.value);
            formData.append('guidance_scale', guidanceSlider.value);
            formData.append('num_images', numImages);
            formData.append('reference_resolution_mode', referenceResolutionSelect?.value || 'balanced_1024');
            formData.append('output_resolution_mode', outputResolutionSelect?.value || 'generated');

            if (stepsInput.value) {
                formData.append('num_inference_steps', stepsInput.value);
            }


            if (seedInput.value) {
                formData.append('seed', seedInput.value);
            }

            // Add LoRA params
            if (loraSelect.value) {
                formData.append('lora_name', loraSelect.value);
                formData.append('lora_scale', loraScaleSlider.value);
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
        stopProgressPolling();

        document.querySelectorAll('.title-text').forEach(s => s.textContent = 'Generated Images');

        // Ensure 100% completion state is visible momentarily before fading out
        ['gen-header-progress', 'gen-header-progress-placeholder'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.classList.remove('initializing');
                el.classList.add('active');
                el.style.width = '100%';

                // Visually satisfying completion flash
                const originalShadow = el.style.boxShadow;
                el.style.boxShadow = '0 0 50px rgba(255,255,255,1), inset 0 0 30px rgba(255,255,255,1)';

                setTimeout(() => {
                    el.classList.remove('active');
                    el.style.boxShadow = originalShadow;
                }, 1500);
            }
        });

        generateBtn.disabled = false;
        genStatus.classList.add('hidden');
    }
}

// =============================================================================
// Gallery Entrance Animations
// =============================================================================

const ENTRANCE_ANIMATIONS = [
    'gallery-anim-cosmic-zoom',
    'gallery-anim-balloon-pop',
    'gallery-anim-slide-left',
    'gallery-anim-slide-right',
    'gallery-anim-drop-bounce',
    'gallery-anim-spiral-in',
    'gallery-anim-glitch',
    'gallery-anim-unfold',
];

// Fisher-Yates shuffle (returns a new array)
function shuffleArray(arr) {
    const shuffled = [...arr];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
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

    // Shuffle animations so each image in a batch gets a different one
    const shuffledAnims = shuffleArray(ENTRANCE_ANIMATIONS);

    images.forEach((imgData, index) => {
        const item = document.createElement('div');
        item.className = 'gallery-item gallery-item-hidden';
        item.style.animationDelay = `${index * 120}ms`;
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

        // Pick a random animation (cycling through shuffled list for variety)
        const animClass = shuffledAnims[index % shuffledAnims.length];

        // Apply the animation after a frame so the hidden state renders first
        requestAnimationFrame(() => {
            item.classList.add(animClass);
        });

        // Clean up animation classes once animation finishes so hover effects work
        item.addEventListener('animationend', () => {
            item.classList.remove('gallery-item-hidden', animClass);
            item.style.animationDelay = '';
        }, { once: true });

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

// Download source image from src URL (blob or data URL)
function downloadSourceImage(src, filename) {
    const link = document.createElement('a');
    link.href = src;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
            <p class="modal-text">Choose the reference role this image should fill.</p>
            <div class="modal-actions slot-modal-grid">
                <button class="btn btn-secondary" id="slot-modal-cancel">Cancel</button>
                <button class="btn btn-primary" data-slot-target="0">Primary</button>
                <button class="btn btn-primary" data-slot-target="1">Character</button>
                <button class="btn btn-primary" data-slot-target="2">Pose</button>
                <button class="btn btn-primary" data-slot-target="3">Style</button>
            </div>
        </div>
    `;
    document.body.appendChild(slotModal);

    const cancelBtn = slotModal.querySelector('#slot-modal-cancel');

    cancelBtn.addEventListener('click', closeSlotModal);
    slotModal.querySelectorAll('[data-slot-target]').forEach(button => {
        button.addEventListener('click', () => transferPendingImageToSlot(Number(button.dataset.slotTarget)));
    });

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

    setReferenceFile(slotNumber, file);

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
renderCharacterProfiles();
updatePromptComposer({ force: true });

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
const videoQualityPresetSelect = document.getElementById('video-quality-preset-select');
const videoFramesSelect = document.getElementById('video-frames-select');
const videoFpsSelect = document.getElementById('video-fps-select');
const videoStepsInput = document.getElementById('video-steps-input');
const videoSeedInput = document.getElementById('video-seed-input');
const videoGuidanceSlider = document.getElementById('video-guidance-slider');
const videoGuidanceValue = document.getElementById('video-guidance-value');
const videoImageFitSelect = document.getElementById('video-image-fit-select');
const videoResolutionSelect = document.getElementById('video-resolution-select');
const videoNegativePromptInput = document.getElementById('video-negative-prompt-input');
const videoMaxSequenceLengthSelect = document.getElementById('video-max-sequence-length-select');
const videoExportQualityInput = document.getElementById('video-export-quality-input');
const videoDecodeTimestepInput = document.getElementById('video-decode-timestep-input');
const videoDecodeNoiseScaleInput = document.getElementById('video-decode-noise-scale-input');
const videoGuidanceRescaleInput = document.getElementById('video-guidance-rescale-input');
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

const VIDEO_QUALITY_PRESETS = {
    fast: { frames: '49', steps: '8', guidance: '1.0' },
    balanced: { frames: '73', steps: '16', guidance: '1.0' },
    quality: { frames: '97', steps: '30', guidance: '1.0' },
    comfy: { frames: '97', steps: '30', guidance: '3.0' },
};

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

if (videoQualityPresetSelect) {
    videoQualityPresetSelect.addEventListener('change', () => {
        const preset = VIDEO_QUALITY_PRESETS[videoQualityPresetSelect.value];
        if (!preset) return;

        videoFramesSelect.value = preset.frames;
        videoStepsInput.value = preset.steps;
        videoGuidanceSlider.value = preset.guidance;
        videoGuidanceValue.textContent = preset.guidance;
    });
}

[videoFramesSelect, videoStepsInput, videoGuidanceSlider].forEach(control => {
    control?.addEventListener('input', () => {
        if (videoQualityPresetSelect) videoQualityPresetSelect.value = 'custom';
    });
    control?.addEventListener('change', () => {
        if (videoQualityPresetSelect) videoQualityPresetSelect.value = 'custom';
    });
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
    videoGenStatusText.textContent = 'Connecting to LTX-Video server...';

    try {
        // Check if server is available
        const healthCheck = await fetch(`${VIDEO_API_BASE}/api/video/health`).catch(() => null);

        if (!healthCheck || !healthCheck.ok) {
            throw new Error('Cannot connect to video server. Make sure video_server.py is running on port 8001.');
        }

        const healthData = await healthCheck.json();
        if (!healthData.model_loaded) {
            videoGenStatusText.textContent = 'Loading LTX-Video 2B model (this may take a few minutes on first run)...';
        } else {
            videoGenStatusText.textContent = 'Generating video (this may take 2-5 minutes)...';
        }

        // Build form data with image for I2V
        const formData = new FormData();
        formData.append('image', videoSourceFile);
        formData.append('prompt', prompt);
        formData.append('num_frames', videoFramesSelect.value);
        formData.append('fps', videoFpsSelect.value);
        formData.append('num_inference_steps', videoStepsInput.value || '8');
        formData.append('guidance_scale', videoGuidanceSlider.value);
        formData.append('quality_preset', videoQualityPresetSelect?.value || 'custom');
        formData.append('image_fit_mode', videoImageFitSelect?.value || 'blur');
        formData.append('resolution_preset', videoResolutionSelect?.value || 'auto');
        formData.append('max_sequence_length', videoMaxSequenceLengthSelect?.value || '128');
        formData.append('export_quality', videoExportQualityInput?.value || '9');
        formData.append('decode_timestep', videoDecodeTimestepInput?.value || '0.05');
        formData.append('decode_noise_scale', videoDecodeNoiseScaleInput?.value || '0.025');
        formData.append('guidance_rescale', videoGuidanceRescaleInput?.value || '0');

        if (videoNegativePromptInput?.value.trim()) {
            formData.append('negative_prompt', videoNegativePromptInput.value.trim());
        }

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
        const videoMeta = [
            `Seed: ${result.seed}`,
            result.width && result.height ? `${result.width} x ${result.height}` : null,
            result.num_frames && result.fps ? `${result.num_frames} frames @ ${result.fps} FPS` : null,
        ].filter(Boolean).join(' | ');
        videoSeedDisplay.textContent = videoMeta;

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
    link.download = `ltx_video_generation_${lastVideoSeed || Date.now()}.mp4`;
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

        startProgressPolling('upscale-progress-fill', 'upscale-progress-text');

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
        stopProgressPolling();
        const textEl = document.getElementById('upscale-progress-text');
        const fillEl = document.getElementById('upscale-progress-fill');
        if (fillEl) fillEl.style.width = '100%';
        if (textEl) textEl.textContent = '[===================>] 100% DONE';

        upscaleBtn.disabled = false;
        upscaleStatus.classList.add('hidden');
    }
}






