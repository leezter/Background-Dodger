# Background-Dodger

A client-side image background removal webapp powered by AI. All processing happens locally in the browser — images never leave your device.

## Quick Start

```bash
# Serve locally (any static file server works)
npx http-server . -p 8080 -o
```

Then open [http://localhost:8080](http://localhost:8080)

## Features

- **AI-Powered**: Uses ONNX neural network models via WebAssembly
- **Privacy-First**: Zero server-side processing, images stay local
- **Original Quality**: Maintains full resolution of uploaded images
- **Drag & Drop**: Intuitive file upload interface
- **PNG Export**: Downloads with transparent background

## Project Structure

```
BackgroundRemover/
├── index.html      # Main HTML structure
├── styles.css      # Dark glassmorphism theme, animations, responsive design
├── app.js          # Application logic, library integration
└── README.md       # This file
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| UI | HTML5, CSS3, Vanilla JavaScript |
| Background Removal | [@imgly/background-removal](https://github.com/imgly/background-removal-js) v1.5.1 |
| Fonts | [Inter](https://fonts.google.com/specimen/Inter) via Google Fonts |

## Usage Notes

- **First Load**: Downloads ~30MB AI model (cached by browser for future use)
- **Supported Formats**: PNG, JPG, JPEG, WebP
- **Output Format**: PNG with transparent background
- **No Internet Required**: After initial model download, works offline

## Browser Compatibility

Requires browsers with WebAssembly and ES Modules support:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+

## License

MIT
