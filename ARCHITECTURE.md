# Architecture & AI Agent Guide

This document is intended for AI assistants and developers working on this codebase.

## System Overview

This app provides **two AI-powered features**, each with a different architecture:

### Background Remover (Browser-based)
```mermaid
flowchart LR
    A[User uploads image] --> B[app.js handles file]
    B --> C[imgly/background-removal library]
    C --> D[ONNX Model via WebAssembly]
    D --> E[Segmentation mask generated]
    E --> F[Background removed]
    F --> G[PNG blob returned]
    G --> H[Display & download available]
```

### FLUX.2 Image Generator (GPU-based)
```mermaid
flowchart LR
    A[User enters prompt] --> B[app.js sends request]
    B --> C[Python FastAPI Server :8000]
    C --> D[Flux2KleinPipeline]
    D --> E[Local GPU RTX 4070]
    E --> F[Image generated]
    F --> G[Base64 PNG returned]
    G --> H[Display & download in browser]
```

### CogVideoX Video Generator (GPU-based)
```mermaid
flowchart LR
    A[User uploads image + prompt] --> B[app.js sends request]
    B --> C[Python FastAPI Server :8001]
    C --> D[CogVideoX-5b-I2V Pipeline]
    D --> E[Local GPU with INT8 quantization]
    E --> F[Video frames generated]
    F --> G[MP4 encoded]
    G --> H[Base64 MP4 returned]
    H --> I[Video player in browser]
```

## Three-Server Architecture

| Server | Port | Purpose | Required For |
|--------|------|---------|--------------|
| http-server | 8080 | Serves frontend HTML/CSS/JS | All features |
| flux_server.py | 8000 | Python API for FLUX.2 GPU inference | Image Generator only |
| video_server.py | 8001 | Python API for CogVideoX I2V | Video Generator only |

## File Responsibilities

### `index.html`
- **Purpose**: DOM structure and layout
- **Key Sections**:
  - `.sidebar`: Left navigation panel with icons for switching features
  - `#bg-remover-panel`: Background Remover UI (upload, processing, result)
  - `#image-gen-panel`: FLUX.2 Image Generator UI (prompt, settings, result)
- **Panel System**: Uses `.panel.active` class for visibility, switched via sidebar nav
- **Dependencies**: Loads `styles.css` and `app.js` (ES module)

### `styles.css`
- **Purpose**: Visual design and responsive layout
- **Design System**: CSS custom properties in `:root` for colors, spacing, radii
- **Key Features**:
  - Glassmorphism effects (`backdrop-filter: blur`)
  - Animated gradient orbs in background
  - Left sidebar with hover-expand labels
  - Checkered pattern for transparency preview
  - Mobile-first responsive breakpoints at 768px and 480px

### `app.js`
- **Purpose**: Application logic and library integration
- **Key Functions**:
  - **Navigation**: Panel switching via `.nav-item` click handlers
  - **Background Remover**: `handleFile()`, `removeBackground()`, `downloadImage()`
  - **Image Generator**: `generateImage()`, mode toggle (text2img/img2img), FLUX API calls
  - `showSection(name)`: Manages UI state transitions
  - `showError(message)`: Toast notification system
- **API Configuration**: `FLUX_API_BASE = 'http://127.0.0.1:8000'`

### `server/` Directory (Python Backend)

#### `server/flux_server.py`
- **Purpose**: FastAPI server for FLUX.2 GPU inference
- **Key Endpoints**:
  - `GET /api/health`: Health check with CUDA status
  - `GET /api/models`: List available models
  - `POST /api/generate`: Text-to-image generation
  - `POST /api/img2img`: Image-to-image editing
  - `POST /api/load-model`: Switch between models
- **Model Loading**: Auto-selects model based on available VRAM
- **Pipeline**: Uses `Flux2KleinPipeline` from diffusers (git main branch)

#### `server/requirements.txt`
Python dependencies:
```
torch>=2.1.0 (with CUDA)
diffusers (from git main)
transformers, accelerate, safetensors
fastapi, uvicorn, python-multipart, pillow
```

### FLUX.2 Models Supported

| Model Key | Hugging Face ID | VRAM | License |
|-----------|-----------------|------|---------|
| `klein-4b` | `black-forest-labs/FLUX.2-klein-4B` | ~13GB | Apache 2.0 |
| `klein-9b` | `black-forest-labs/FLUX.2-klein-9B` | ~20GB | Non-commercial |
| `dev` | `black-forest-labs/FLUX.2-dev` | ~80GB | Non-commercial |

## External Dependencies

### @imgly/background-removal

**Import Method**: ESM via jsDelivr CDN
```javascript
import { removeBackground as imglyRemoveBackground } from 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.1/+esm';
```

**Library Behavior**:
1. First call downloads ONNX model files (~30MB total)
2. Models cached in browser's Cache Storage
3. Runs inference via WebAssembly (no GPU required, but slower)
4. Returns a `Blob` with transparent PNG data

**Progress Callback**:
```javascript
const config = {
    progress: (key, current, total) => {
        // key: 'fetch:*' for downloads, 'compute:*' for processing
        // current/total: progress numerator/denominator
    }
};
await imglyRemoveBackground(file, config);
```

## State Management

Simple state via module-level variables:
- `processedBlob`: Stores result for download
- `originalFileName`: Used in download filename

UI state managed by showing/hiding sections via `.hidden` class.

---

## Scaling & Improvement Opportunities

### Performance Enhancements

1. **Web Worker Processing**
   - Move `imglyRemoveBackground` call to a Web Worker
   - Prevents UI freezing during large image processing
   - Library supports worker mode via config option

2. **WebGPU Acceleration**
   - Library has experimental WebGPU support
   - Significantly faster on supported browsers
   - Enable via config: `{ device: 'gpu' }`

3. **Image Compression**
   - Add option to compress output PNG
   - Use canvas API for JPEG export with quality slider

### Feature Additions

1. **Batch Processing**
   - Allow multiple file selection
   - Queue system with progress per file
   - Zip download for results

2. **Background Replacement**
   - Instead of transparent, allow solid color or image background
   - Use canvas compositing to layer processed image over new background

3. **Edge Refinement**
   - Add feathering/smoothing slider for edges
   - Post-process mask with blur before applying

4. **Crop & Resize**
   - Add pre-processing crop tool
   - Resize output option (for social media sizes)

5. **Undo/History**
   - Store processing states
   - Allow reverting to previous versions

### Infrastructure

1. **PWA Conversion**
   - Add `manifest.json` and service worker
   - Enable "Add to Home Screen"
   - Full offline support after initial model download

2. **Build System**
   - Add Vite or similar for:
     - TypeScript support
     - Bundle optimization
     - Environment variables

3. **Testing**
   - Add Playwright/Cypress for E2E tests
   - Test various image formats and sizes
   - Test error handling paths

### Alternative Libraries

If `@imgly/background-removal` doesn't meet needs:

| Library | Pros | Cons |
|---------|------|------|
| [rembg](https://github.com/danielgatis/rembg) | Better quality for some images | Requires Python backend |
| [Remove.bg API](https://www.remove.bg/api) | Professional quality | Paid API, not client-side |
| [Carvekit](https://github.com/OPHoperHPO/image-background-remove-tool) | Multiple model options | Python backend required |

---

## Common Issues

### "imglyRemoveBackground is not a function"
- **Cause**: Import syntax issue or naming conflict
- **Fix**: Use named import with alias: `import { removeBackground as imglyRemoveBackground }`

### Model download fails
- **Cause**: CORS issues or network problems
- **Fix**: Ensure running via HTTP server (not `file://` protocol)

### Processing very slow
- **Cause**: Large images + CPU-only processing
- **Fix**: Consider resizing images before processing, or enable WebGPU if available

### Memory errors on large images
- **Cause**: Browser memory limits
- **Fix**: Implement image size limits or chunked processing
