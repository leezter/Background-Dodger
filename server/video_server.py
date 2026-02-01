"""
LTX-Video Generation Server (2B Distilled)

A FastAPI server for local Lightricks LTX-Video generation using the 2B distilled model.

Model: Lightricks/LTX-Video (2B Distilled)
- Storage: ~6.3 GB download
- VRAM: ~8-10GB with optimizations
- Output: Variable frames at 24fps
- Steps: 8 (distilled model optimized for fast generation)
- Compatible with 12GB VRAM + 32GB RAM
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

# Model configuration - LTX-Video 2B Distilled
# Optimized model that fits comfortably in 12GB VRAM + 32GB RAM
LTX_MODEL = {
    "repo_id": "Lightricks/LTX-Video",
    "ckpt_path": "ltxv-2b-0.9.8-distilled.safetensors",
    "name": "LTX-Video 2B Distilled",
    "description": "Image-to-Video generation (2B Distilled, 8 steps)",
    "default_num_frames": 97,  # ~4 seconds at 24fps
    "fps": 24,
    "default_steps": 8,  # Distilled model requires only 8 steps
    "max_steps": 20,  # Distilled model doesn't benefit from many steps
    "default_guidance": 1.0  # Distilled model uses guidance_scale=1.0
}

# =============================================================================
# Model Loading
# =============================================================================

def load_video_model():
    """Load LTX-Video 2B Distilled model optimized for 12GB VRAM + 32GB RAM."""
    if video_state.model_loaded and video_state.pipeline is not None:
        logger.info("LTX-Video model already loaded")
        return
    
    logger.info("Loading LTX-Video 2B Distilled model...")
    logger.info(f"Device: {video_state.device}, dtype: {video_state.dtype}")
    
    try:
        import gc
        from diffusers import LTXImageToVideoPipeline
        from transformers import T5EncoderModel, T5Tokenizer
        from huggingface_hub import hf_hub_download
        
        repo_id = LTX_MODEL["repo_id"]
        ckpt_path = LTX_MODEL["ckpt_path"]
        
        # Download the checkpoint file first
        logger.info(f"Downloading LTX-Video checkpoint from {repo_id}/{ckpt_path}...")
        logger.info("First run will download ~6.3GB model + ~9.4GB T5 text encoder")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_path,
        )
        logger.info(f"Checkpoint downloaded to: {local_path}")
        
        # Load T5 text encoder with memory-efficient loading
        # Use device_map="auto" to load directly to GPU and avoid RAM spike
        logger.info("Loading T5 text encoder (memory-efficient mode)...")
        text_encoder = T5EncoderModel.from_pretrained(
            "Lightricks/LTX-Video-0.9.5",
            subfolder="text_encoder",
            torch_dtype=video_state.dtype,
            device_map="auto",  # Load directly to GPU
            low_cpu_mem_usage=True,  # Minimize RAM usage during loading
        )
        tokenizer = T5Tokenizer.from_pretrained(
            "Lightricks/LTX-Video-0.9.5",
            subfolder="tokenizer",
        )
        logger.info("T5 text encoder loaded")
        gc.collect()
        
        # Load the 2B distilled model from single file with text encoder
        logger.info("Loading pipeline from checkpoint...")
        video_state.pipeline = LTXImageToVideoPipeline.from_single_file(
            local_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=video_state.dtype,
        )
        
        # Move to GPU
        video_state.pipeline.to(video_state.device)
        
        # Enable VAE tiling and slicing for memory efficiency
        if hasattr(video_state.pipeline, 'vae') and video_state.pipeline.vae is not None:
            video_state.pipeline.vae.enable_tiling()
            video_state.pipeline.vae.enable_slicing()
            logger.info("VAE tiling and slicing enabled")
            
        video_state.model_loaded = True
        logger.info("LTX-Video 2B Distilled model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load LTX-Video model: {e}")
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
    logger.info("Starting LTX-Video 2B Generation Server...")
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
    logger.info("Shutting down LTX-Video server...")

app = FastAPI(
    title="LTX-Video Generation API",
    description="Local Lightricks LTX-Video (2B Distilled) generation server",
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
        "model": LTX_MODEL["name"] if video_state.model_loaded else None
    }

@app.post("/api/video/generate", response_model=VideoGenerateResponse)
async def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_frames: int = Form(97),
    fps: int = Form(24),
    num_inference_steps: int = Form(8),
    guidance_scale: float = Form(1.0),
    seed: Optional[int] = Form(None)
):
    """Generate a video from an image and text prompt using LTX-Video 2B Distilled."""
    
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
        
        # LTX-Video works best with resolutions divisible by 32, e.g., 768x512, 1024x576
        # We will target a reasonable default like 768x512 for VRAM usage, or keep native if possible
        # Default target: 768x512 (landscape) or 512x768 (portrait) based on input aspect ratio
        
        width, height = input_image.size
        
        # Simple logic: resizing to nearest multiple of 32
        # Target 512px on the shortest side roughly?
        # LTX usually generates 768x512
        
        target_width = 768
        target_height = 512
        
        # Adjust for portrait if needed
        if height > width:
            target_width, target_height = 512, 768
            
        # Resize to fill
        ratio = max(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        input_image = input_image.crop((left, top, right, bottom))
        
        # Ensure dimensions are divisible by 32
        w, h = input_image.size
        w = w - (w % 32)
        h = h - (h % 32)
        if w != input_image.size[0] or h != input_image.size[1]:
            input_image = input_image.resize((w, h), Image.LANCZOS)
            
        logger.info(f"Prepared input image: {input_image.size[0]}x{input_image.size[1]}")
        
        # Clamp parameters for distilled model
        num_frames = max(9, min(num_frames, 161))  # Cap at 161 for 12GB VRAM
        num_inference_steps = max(4, min(num_inference_steps, LTX_MODEL["max_steps"]))
        
        # Dev model works best with guidance_scale around 3.0
        if guidance_scale < 2.0:
            logger.warning(f"Dev model works best with guidance_scale~3.0, got {guidance_scale}")
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        logger.info(f"Generating video: prompt='{prompt[:50]}...', frames={num_frames}, steps={num_inference_steps}, seed={seed}")
        
        # Generate video with LTX-Video 2B Distilled parameters
        result = video_state.pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=input_image.size[1],
            width=input_image.size[0],
            output_type="pil",
        )
        
        # Get frames from result  
        # LTX-Video returns frames directly as a list of PIL Images
        if hasattr(result, 'frames'):
            frames = result.frames[0] if isinstance(result.frames[0], list) else result.frames
        else:
            frames = result[0]  # Tuple return format
        
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
