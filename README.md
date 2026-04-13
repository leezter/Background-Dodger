# Background-Dodger

A multi-purpose AI toolkit: **Background Remover** (browser-based), **FLUX.2 Image Generator** (local GPU), **LTX-Video Generator** (local GPU), and **Image Upscaler** (local GPU). All processing happens locally — your data never leaves your device.

## Quick Start

### Background Remover Only
```bash
npx http-server . -p 8080 -o
```
Then open [http://localhost:8080](http://localhost:8080)

### With FLUX.2 Image Generator (requires NVIDIA GPU)
```bash
# Terminal 1: Frontend
npx http-server . -p 8080

# Terminal 2: Backend (first time setup)
cd server
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers.git
# Manual install for upscaler dependencies
pip install basicsr
python -m pip install realesrgan opencv-python
python flux_server.py
```
Then open [http://localhost:8080](http://localhost:8080) and click the ⚡ icon (Generator) or 🔍 icon (Upscaler) in the sidebar.

### With LTX-Video Generator (requires NVIDIA GPU)
```bash
# Terminal 1: Frontend
npx http-server . -p 8080

# Terminal 2: Video Backend
cd server
pip install -r requirements.txt
python video_server.py
```
Then open [http://localhost:8080](http://localhost:8080) and click the 🎬 icon in the sidebar.

## Features

### Background Remover
- **AI-Powered**: Uses ONNX neural network models via WebAssembly
- **Privacy-First**: Zero server-side processing, images stay local
- **Original Quality**: Maintains full resolution of uploaded images
- **Drag & Drop**: Intuitive file upload interface
- **PNG Export**: Downloads with transparent background

### FLUX.2 Image Generator
- **State-of-the-Art**: Uses FLUX.2 Klein 4B (Black Forest Labs)
- **Local GPU**: Runs entirely on your NVIDIA GPU (RTX 3090/4070+)
- **Text-to-Image**: Generate images from text prompts
- **Image-to-Image**: Edit existing images with prompts
- **Dual-Image Support**: Combine two source images for image-to-image generation
- **Sub-Second**: ~0.5s generation time on modern GPUs
- **Apache 2.0**: Klein 4B model is commercially usable

### Image Upscaler
- **Real-ESRGAN**: 4x resolution enhancement with high fidelity
- **Face Enhancement**: Optional GFPGAN integration (planned) for face restoration
- **Local Processing**: Runs on the same GPU server as FLUX.2
- **Fallback**: Automatically uses tiled processing if VRAM is limited

### LTX-Video Generator
- **Image-to-Video**: Animate still images with text prompts
- **Local GPU**: Runs on your NVIDIA GPU with CPU offload and VAE tiling/slicing
- **Multiple Orientations**: Uses 12GB-friendly landscape, portrait, and square LTX resolutions
- **Configurable**: Adjust frames (49-161), FPS (12-24), steps, and guidance
- **MP4 Export**: Downloads as standard MP4 video

## Project Structure

```
BackgroundRemover/
├── index.html          # Main HTML with sidebar navigation
├── styles.css          # Dark glassmorphism theme, responsive design
├── app.js              # Frontend logic for all features
├── manifest.json       # PWA manifest
├── sw.js               # Service worker for offline support
├── server/
│   ├── flux_server.py  # FastAPI backend for FLUX.2
│   ├── video_server.py # FastAPI backend for LTX-Video
│   └── requirements.txt
├── ARCHITECTURE.md     # Detailed technical docs for AI assistants
└── README.md           # This file
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| UI | HTML5, CSS3, Vanilla JavaScript |
| Background Removal | [@imgly/background-removal](https://github.com/imgly/background-removal-js) v1.5.1 |
| Image Generation | [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| Image Upscaling | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| Video Generation | [LTX-Video 2B Distilled](https://huggingface.co/Lightricks/LTX-Video) |
| Backend | FastAPI, PyTorch, Diffusers, TorchAO |
| Fonts | [Inter](https://fonts.google.com/specimen/Inter) via Google Fonts |

## GPU Requirements

### Image Generator
| GPU | VRAM | Model | Speed |
|-----|------|-------|-------|
| RTX 3090/4070 | 12-13GB | Klein 4B | Sub-second |
| RTX 4080/4090 | 16-24GB | Klein 4B/9B | Sub-second |
| H100 | 80GB | Dev 32B | ~10s |

### Video Generator
| GPU | VRAM | Model | Notes |
|-----|------|-------|-------|
| RTX 4070+ | 12-16GB | LTX-Video 2B Distilled | With CPU offload |
| RTX 4090 | 24GB | LTX-Video 2B Distilled / larger LTX variants | More headroom |

## Usage Notes

- **First Load**: Downloads ~30MB AI model for background remover (cached)
- **First FLUX Run**: Downloads ~16GB model (cached in HuggingFace cache)
- **First LTX-Video Run**: Downloads the 2B distilled checkpoint and text encoder (cached in HuggingFace cache)
- **Supported Formats**: PNG, JPG, JPEG, WebP
- **Output Format**: PNG (images), MP4 (video)
- **Offline**: All features work offline after initial model downloads

## Browser Compatibility

Requires browsers with WebAssembly and ES Modules support:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+

## License

MIT
