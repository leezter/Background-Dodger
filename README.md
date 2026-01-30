# Background-Dodger

A dual-purpose AI image tool: **Background Remover** (browser-based) and **FLUX.2 Image Generator** (local GPU). All processing happens locally — images never leave your device.

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
python flux_server.py
```
Then open [http://localhost:8080](http://localhost:8080) and click the ⚡ icon in the sidebar.

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
- **Sub-Second**: ~0.5s generation time on modern GPUs
- **Apache 2.0**: Klein 4B model is commercially usable

## Project Structure

```
BackgroundRemover/
├── index.html          # Main HTML with sidebar navigation
├── styles.css          # Dark glassmorphism theme, responsive design
├── app.js              # Frontend logic for both features
├── manifest.json       # PWA manifest
├── sw.js               # Service worker for offline support
├── server/
│   ├── flux_server.py  # FastAPI backend for FLUX.2
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
| Backend | FastAPI, PyTorch, Diffusers |
| Fonts | [Inter](https://fonts.google.com/specimen/Inter) via Google Fonts |

## GPU Requirements (Image Generator)

| GPU | VRAM | Model | Speed |
|-----|------|-------|-------|
| RTX 3090/4070 | 12-13GB | Klein 4B | Sub-second |
| RTX 4080/4090 | 16-24GB | Klein 4B/9B | Sub-second |
| H100 | 80GB | Dev 32B | ~10s |

## Usage Notes

- **First Load**: Downloads ~30MB AI model for background remover (cached)
- **First FLUX Run**: Downloads ~16GB model (cached in HuggingFace cache)
- **Supported Formats**: PNG, JPG, JPEG, WebP
- **Output Format**: PNG
- **Offline**: Both features work offline after initial model downloads

## Browser Compatibility

Requires browsers with WebAssembly and ES Modules support:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+

## License

MIT
