"""
CogVideoX Video Generation Server

A FastAPI server for local CogVideoX-5b-I2V image-to-video generation.
Uses INT8 quantization via torchao for reduced VRAM usage.

Model: CogVideoX-5b-I2V
- Storage: ~20 GB total download
- VRAM: ~12-16 GB with INT8 quantization + optimizations
- Output: 49 frames at 8fps (~6 seconds video)
"""

import io
import os
import base64
import logging
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global State
# =============================================================================

class VideoModelState:
    """Holds the currently loaded video model state."""
    def __init__(self):
        self.pipeline = None
        self.model_loaded: bool = False
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model_path: Optional[str] = None

video_state = VideoModelState()

# Model configuration - CogVideoX-5b-I2V with INT8 quantization
COGVIDEO_MODEL = {
    "repo_id": "THUDM/CogVideoX-5b-I2V",
    "name": "CogVideoX 5B I2V",
    "description": "Image-to-Video generation with INT8 quantization, ~12-16GB VRAM",
    "default_num_frames": 49,  # ~6 seconds at 8fps
    "fps": 8,
    "default_steps": 50,
    "max_steps": 100,
    "default_guidance": 6.0
}

# =============================================================================
# Model Loading
# =============================================================================

def load_video_model():
    """Load CogVideoX-5b-I2V model with INT8 quantization for reduced VRAM."""
    if video_state.model_loaded and video_state.pipeline is not None:
        logger.info("CogVideoX model already loaded")
        return
    
    logger.info("Loading CogVideoX-5b-I2V model with INT8 quantization...")
    logger.info(f"Device: {video_state.device}, dtype: {video_state.dtype}")
    
    try:
        from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
        from transformers import T5EncoderModel
        
        # Try to use torchao for INT8 quantization (if available)
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            use_quantization = True
            quant_config = Int8WeightOnlyConfig()
            logger.info("Using torchao INT8 quantization (Int8WeightOnlyConfig) for reduced VRAM")
        except ImportError:
            # Fallback for older torchao versions or if import fails
            try:
                from torchao.quantization import quantize_, int8_weight_only
                use_quantization = True
                quant_config = int8_weight_only()
                logger.info("Using torchao INT8 quantization (int8_weight_only) for reduced VRAM")
            except ImportError:
                use_quantization = False
                logger.warning("torchao not available, loading without quantization (higher VRAM usage)")
        
        repo_id = COGVIDEO_MODEL["repo_id"]
        
        # Load components separately for optional quantization
        logger.info("Loading text encoder...")
        text_encoder = T5EncoderModel.from_pretrained(
            repo_id, subfolder="text_encoder", torch_dtype=video_state.dtype
        )
        if use_quantization:
            quantize_(text_encoder, quant_config)
        
        logger.info("Loading transformer...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            repo_id, subfolder="transformer", torch_dtype=video_state.dtype
        )
        if use_quantization:
            quantize_(transformer, quant_config)
        
        logger.info("Loading VAE...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            repo_id, subfolder="vae", torch_dtype=video_state.dtype
        )
        if use_quantization:
            quantize_(vae, quant_config)
        
        # Create pipeline with quantized components
        logger.info("Creating pipeline...")
        video_state.pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
            repo_id,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=video_state.dtype,
        )
        
        # Enable memory optimizations
        video_state.pipeline.enable_model_cpu_offload()
        video_state.pipeline.vae.enable_slicing()
        video_state.pipeline.vae.enable_tiling()
        
        video_state.model_loaded = True
        logger.info("CogVideoX-5b-I2V model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load CogVideoX model: {e}")
        raise

# =============================================================================
# Pydantic Models
# =============================================================================

class VideoGenerateResponse(BaseModel):
    success: bool
    video: Optional[str] = None  # Base64 encoded MP4
    error: Optional[str] = None
    seed: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = None

# =============================================================================
# Video Export Utility
# =============================================================================

def export_to_video(frames, fps: int = 8) -> bytes:
    """Export PIL Image frames to MP4 video bytes."""
    import imageio
    
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Convert frames to numpy arrays
        import numpy as np
        frame_arrays = [np.array(frame) for frame in frames]
        
        # Write video
        writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264', quality=8)
        for frame in frame_arrays:
            writer.append_data(frame)
        writer.close()
        
        # Read the video bytes
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        return video_bytes
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    logger.info("Starting CogVideoX Video Generation Server...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {vram_gb:.1f} GB")
    
    # Load model on startup
    try:
        load_video_model()
    except Exception as e:
        logger.warning(f"Could not load video model on startup: {e}")
        logger.info("Model will be loaded on first generation request")
    
    yield
    
    # Cleanup
    logger.info("Shutting down CogVideoX server...")

app = FastAPI(
    title="CogVideoX Video Generation API",
    description="Local CogVideoX-1.5-I2V image-to-video generation server",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/video/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": video_state.model_loaded,
        "device": video_state.device,
        "model": COGVIDEO_MODEL["name"] if video_state.model_loaded else None
    }

@app.post("/api/video/generate", response_model=VideoGenerateResponse)
async def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_frames: int = Form(49),
    fps: int = Form(8),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(6.0),
    seed: Optional[int] = Form(None)
):
    """Generate a video from an image and text prompt using CogVideoX-5b-I2V."""
    
    # Ensure model is loaded
    if not video_state.model_loaded or video_state.pipeline is None:
        try:
            load_video_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {str(e)}")
    
    try:
        # Load the input image
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # CogVideoX supports multiple resolutions
        # Determine best target based on aspect ratio
        width, height = input_image.size
        aspect_ratio = width / height
        
        # Choose target dimensions based on aspect ratio
        if 0.9 <= aspect_ratio <= 1.1:
            # Square-ish image -> use 480x480
            target_width, target_height = 480, 480
        elif aspect_ratio > 1.1:
            # Landscape image -> use 720x480
            target_width, target_height = 720, 480
        else:
            # Portrait image -> use 480x720
            target_width, target_height = 480, 720
        
        if width != target_width or height != target_height:
            input_image = input_image.resize((target_width, target_height), Image.LANCZOS)
            logger.info(f"Resized input image from {width}x{height} to {target_width}x{target_height}")
        
        # Clamp parameters
        num_frames = max(9, min(num_frames, 49))  # CogVideoX-5b-I2V limit
        num_inference_steps = max(10, min(num_inference_steps, COGVIDEO_MODEL["max_steps"]))
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        logger.info(f"Generating video: prompt='{prompt[:50]}...', frames={num_frames}, steps={num_inference_steps}, seed={seed}")
        
        # Generate video (image-to-video)
        result = video_state.pipeline(
            image=input_image,
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Get frames from result
        frames = result.frames[0]  # List of PIL Images
        
        # Export to MP4 with user-specified FPS
        fps = max(1, min(fps, 30))  # Clamp to reasonable range
        video_bytes = export_to_video(frames, fps=fps)
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        
        logger.info(f"Video generated successfully: {len(frames)} frames, {len(video_bytes)} bytes")
        
        return VideoGenerateResponse(
            success=True,
            video=video_base64,
            seed=seed,
            num_frames=len(frames),
            fps=fps
        )
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return VideoGenerateResponse(
            success=False,
            error=str(e)
        )

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
