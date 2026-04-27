"""
LTX-Video Generation Server (2B Distilled)

A FastAPI server for local Lightricks LTX-Video generation using the 2B distilled model.

Model: Lightricks/LTX-Video (2B Distilled)
- Storage: ~6.3 GB checkpoint download, plus shared text encoder cache
- VRAM: 12GB target with CPU offload
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
from PIL import Image, ImageFilter, ImageOps
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
# Restored fast model that fits comfortably in 12GB VRAM + 32GB RAM.
LTX_MODEL = {
    "repo_id": "Lightricks/LTX-Video",
    "ckpt_path": "ltxv-2b-0.9.8-distilled.safetensors",
    "text_encoder_repo_id": "Lightricks/LTX-Video-0.9.5",
    "name": "LTX-Video 2B Distilled",
    "description": "Image-to-Video generation (2B Distilled, optimized for 12GB VRAM)",
    "default_num_frames": 49,  # Shorter clips drift/morph less on I2V.
    "fps": 24,
    "default_steps": 8,  # Distilled model requires only 8 steps for fast generation.
    "max_steps": 30,
    "default_guidance": 1.0,
    "max_guidance": 6.0,
    "landscape_size": (768, 512),
    "wide_size": (832, 480),
    "diffusers_size": (704, 480),
    "portrait_size": (512, 768),
    "square_size": (640, 640),
    "preview_landscape_size": (512, 352),
    "preview_medium_landscape_size": (576, 384),
    "preview_wide_size": (640, 384),
    "preview_portrait_size": (352, 512),
    "preview_medium_portrait_size": (384, 576),
    "preview_square_size": (512, 512),
    "decode_timestep": 0.05,
    "decode_noise_scale": 0.025,
    "guidance_rescale": 0.0,
    "max_sequence_length": 128,
    "export_quality": 9,
    "negative_prompt": (
        "worst quality, low quality, deformed, distorted, disfigured, morphing, identity shift, face changing, "
        "body changing, object warping, inconsistent motion, motion smear, motion artifacts, blurry, jittery, "
        "fused fingers, bad anatomy, weird hands, extra limbs, new objects appearing, scene change"
    ),
    "quality_presets": {
        "fast": {"num_frames": 49, "num_inference_steps": 8, "guidance_scale": 1.0},
        "balanced": {"num_frames": 73, "num_inference_steps": 16, "guidance_scale": 1.0},
        "quality": {"num_frames": 97, "num_inference_steps": 30, "guidance_scale": 1.0},
        "comfy": {"num_frames": 97, "num_inference_steps": 30, "guidance_scale": 3.0},
    },
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
        from transformers import T5EncoderModel, T5TokenizerFast
        from huggingface_hub import hf_hub_download
        
        repo_id = LTX_MODEL["repo_id"]
        ckpt_path = LTX_MODEL["ckpt_path"]
        text_encoder_repo_id = LTX_MODEL["text_encoder_repo_id"]
        
        # Download the checkpoint file first
        logger.info(f"Downloading LTX-Video checkpoint from {repo_id}/{ckpt_path}...")
        logger.info("First run will download ~6.3GB checkpoint + shared T5 text encoder cache")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_path,
        )
        logger.info(f"Checkpoint downloaded to: {local_path}")
        
        # Load T5 text encoder with memory-efficient loading
        logger.info("Loading T5 text encoder (memory-efficient mode)...")
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_repo_id,
            subfolder="text_encoder",
            torch_dtype=video_state.dtype,
            low_cpu_mem_usage=True,  # Minimize RAM usage during loading
        )
        tokenizer = T5TokenizerFast.from_pretrained(
            text_encoder_repo_id,
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
        
        # Keep VRAM headroom on 12GB GPUs. This is slower than moving the whole
        # pipeline to CUDA, but quality is unchanged and OOM risk is much lower.
        if video_state.device == "cuda":
            video_state.pipeline.enable_model_cpu_offload()
            logger.info("Model CPU offload enabled")
        else:
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
    width: Optional[int] = None
    height: Optional[int] = None

# =============================================================================
# Video Utilities
# =============================================================================

def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def normalize_ltx_frame_count(num_frames: int) -> int:
    """LTX works best with frame counts of 8n + 1."""
    num_frames = _clamp_int(num_frames, 9, 161)
    return ((num_frames - 1) // 8) * 8 + 1


def choose_ltx_size(width: int, height: int, resolution_preset: str = "auto") -> tuple[int, int]:
    """Choose a 12GB-friendly target size while preserving the input orientation."""
    preset_sizes = {
        "landscape_768x512": LTX_MODEL["landscape_size"],
        "wide_832x480": LTX_MODEL["wide_size"],
        "diffusers_704x480": LTX_MODEL["diffusers_size"],
        "portrait_512x768": LTX_MODEL["portrait_size"],
        "square_640x640": LTX_MODEL["square_size"],
        "preview_landscape_512x352": LTX_MODEL["preview_landscape_size"],
        "preview_landscape_576x384": LTX_MODEL["preview_medium_landscape_size"],
        "preview_wide_640x384": LTX_MODEL["preview_wide_size"],
        "preview_portrait_352x512": LTX_MODEL["preview_portrait_size"],
        "preview_portrait_384x576": LTX_MODEL["preview_medium_portrait_size"],
        "preview_square_512x512": LTX_MODEL["preview_square_size"],
    }
    if resolution_preset in preset_sizes:
        return preset_sizes[resolution_preset]

    aspect_ratio = width / height
    if resolution_preset == "auto_preview":
        if aspect_ratio > 1.45:
            return LTX_MODEL["preview_wide_size"]
        if aspect_ratio > 1.15:
            return LTX_MODEL["preview_medium_landscape_size"]
        if aspect_ratio < 0.87:
            return LTX_MODEL["preview_medium_portrait_size"]
        return LTX_MODEL["preview_square_size"]

    if aspect_ratio > 1.15:
        return LTX_MODEL["landscape_size"]
    if aspect_ratio < 0.87:
        return LTX_MODEL["portrait_size"]
    return LTX_MODEL["square_size"]


def prepare_image_for_ltx(
    input_image: Image.Image,
    image_fit_mode: str = "blur",
    resolution_preset: str = "auto",
) -> Image.Image:
    """Resize for LTX without cutting off the source image content."""
    original_width, original_height = input_image.size
    target_width, target_height = choose_ltx_size(original_width, original_height, resolution_preset)
    target_size = (target_width, target_height)

    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    if abs(original_ratio - target_ratio) < 0.02:
        return input_image.resize(target_size, Image.LANCZOS)

    if image_fit_mode == "crop":
        return ImageOps.fit(
            input_image,
            target_size,
            method=Image.LANCZOS,
            centering=(0.5, 0.5),
        )

    if image_fit_mode == "pad":
        background = Image.new("RGB", target_size, (16, 16, 16))
        foreground = input_image.copy()
        foreground.thumbnail(target_size, Image.LANCZOS)
        left = (target_width - foreground.size[0]) // 2
        top = (target_height - foreground.size[1]) // 2
        background.paste(foreground, (left, top))
        return background

    background = ImageOps.fit(
        input_image,
        target_size,
        method=Image.LANCZOS,
        centering=(0.5, 0.5),
    ).filter(ImageFilter.GaussianBlur(radius=24))

    foreground = input_image.copy()
    foreground.thumbnail(target_size, Image.LANCZOS)
    left = (target_width - foreground.size[0]) // 2
    top = (target_height - foreground.size[1]) // 2
    background.paste(foreground, (left, top))
    return background


def apply_quality_preset(
    quality_preset: str,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> tuple[int, int, float]:
    preset = LTX_MODEL["quality_presets"].get(quality_preset)
    if not preset:
        return num_frames, num_inference_steps, guidance_scale

    return (
        preset["num_frames"],
        preset["num_inference_steps"],
        preset["guidance_scale"],
    )

def export_to_video(frames, fps: int = 8, quality: int = 8) -> bytes:
    """Export PIL Image frames to MP4 video bytes."""
    import imageio
    
    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Convert frames to numpy arrays
        import numpy as np
        frame_arrays = [np.array(frame.convert("RGB")) for frame in frames]
        
        # Write video
        writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264', quality=quality)
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
        "model": LTX_MODEL["name"] if video_state.model_loaded else None,
        "defaults": {
            "num_frames": LTX_MODEL["default_num_frames"],
            "fps": LTX_MODEL["fps"],
            "num_inference_steps": LTX_MODEL["default_steps"],
            "guidance_scale": LTX_MODEL["default_guidance"],
            "guidance_rescale": LTX_MODEL["guidance_rescale"],
            "decode_timestep": LTX_MODEL["decode_timestep"],
            "decode_noise_scale": LTX_MODEL["decode_noise_scale"],
            "max_sequence_length": LTX_MODEL["max_sequence_length"],
            "export_quality": LTX_MODEL["export_quality"],
        },
    }

@app.post("/api/video/generate", response_model=VideoGenerateResponse)
def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    num_frames: int = Form(49),
    fps: int = Form(24),
    num_inference_steps: int = Form(8),
    guidance_scale: float = Form(1.0),
    seed: Optional[int] = Form(None),
    quality_preset: str = Form("custom"),
    negative_prompt: Optional[str] = Form(None),
    image_fit_mode: str = Form("blur"),
    resolution_preset: str = Form("auto"),
    guidance_rescale: float = Form(0.0),
    decode_timestep: float = Form(0.05),
    decode_noise_scale: float = Form(0.025),
    max_sequence_length: int = Form(128),
    export_quality: int = Form(9),
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
        image_data = image.file.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        width, height = input_image.size
        image_fit_mode = image_fit_mode if image_fit_mode in {"blur", "crop", "pad"} else "blur"
        resolution_preset = resolution_preset if resolution_preset in {
            "auto",
            "auto_preview",
            "landscape_768x512",
            "wide_832x480",
            "diffusers_704x480",
            "portrait_512x768",
            "square_640x640",
            "preview_landscape_512x352",
            "preview_landscape_576x384",
            "preview_wide_640x384",
            "preview_portrait_352x512",
            "preview_portrait_384x576",
            "preview_square_512x512",
        } else "auto"

        input_image = prepare_image_for_ltx(input_image, image_fit_mode, resolution_preset)
        prepared_width, prepared_height = input_image.size

        logger.info(
            "Prepared input image: original %sx%s -> LTX %sx%s (%s, %s)",
            width,
            height,
            prepared_width,
            prepared_height,
            image_fit_mode,
            resolution_preset,
        )
        
        # Clamp parameters for the distilled model
        num_frames, num_inference_steps, guidance_scale = apply_quality_preset(
            quality_preset,
            num_frames,
            num_inference_steps,
            guidance_scale,
        )
        num_frames = normalize_ltx_frame_count(num_frames)
        num_inference_steps = max(4, min(num_inference_steps, LTX_MODEL["max_steps"]))
        fps = _clamp_int(fps, 1, 30)
        guidance_scale = max(1.0, min(guidance_scale, LTX_MODEL["max_guidance"]))
        guidance_rescale = max(0.0, min(guidance_rescale, 1.0))
        decode_timestep = max(0.0, min(decode_timestep, 0.2))
        decode_noise_scale = max(0.0, min(decode_noise_scale, 0.2))
        max_sequence_length = _clamp_int(max_sequence_length, 64, 256)
        export_quality = _clamp_int(export_quality, 5, 10)
        negative_prompt = (negative_prompt or LTX_MODEL["negative_prompt"]).strip() or LTX_MODEL["negative_prompt"]
        
        if guidance_scale > 2.0:
            logger.warning(
                "LTX 2B distilled is tuned for guidance_scale=1.0; high CFG may reduce visual quality. got=%s",
                guidance_scale,
            )
        
        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=video_state.device).manual_seed(seed)
        
        logger.info(f"Generating video: prompt='{prompt[:50]}...', frames={num_frames}, steps={num_inference_steps}, seed={seed}")

        # Generate video with LTX-Video 2B Distilled parameters
        with torch.inference_mode():
            result = video_state.pipeline(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                generator=generator,
                height=prepared_height,
                width=prepared_width,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                output_type="pil",
                max_sequence_length=max_sequence_length,
            )
        
        # Get frames from result  
        # LTX-Video returns frames directly as a list of PIL Images
        if hasattr(result, 'frames'):
            frames = result.frames[0] if isinstance(result.frames[0], list) else result.frames
        else:
            frames = result[0]  # Tuple return format
        
        # Export to MP4 with the same FPS used for temporal conditioning.
        video_bytes = export_to_video(frames, fps=fps, quality=export_quality)
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        
        logger.info(f"Video generated successfully: {len(frames)} frames, {len(video_bytes)} bytes")
        
        return VideoGenerateResponse(
            success=True,
            video=video_base64,
            seed=seed,
            num_frames=len(frames),
            fps=fps,
            width=prepared_width,
            height=prepared_height,
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
