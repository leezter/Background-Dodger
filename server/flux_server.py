"""
FLUX.2 Image Generation Server

A FastAPI server for local FLUX.2 image generation using GPU acceleration.
Supports text-to-image and image-to-image generation with model switching.

FLUX.2 Models:
- FLUX.2 [klein] 4B: Fast, Apache 2.0, ~13GB VRAM, sub-second generation
- FLUX.2 [klein] 9B: Higher quality, non-commercial license
- FLUX.2 [dev]: 32B parameters, highest quality (requires H100-level GPU)
"""

import io
import base64
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global State
# =============================================================================

class ModelState:
    """Holds the currently loaded model state."""
    def __init__(self):
        self.pipeline = None
        self.current_model: Optional[str] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model_state = ModelState()

# Available FLUX.2 models
AVAILABLE_MODELS = {
    "klein-4b": {
        "id": "black-forest-labs/FLUX.2-klein-4B",
        "name": "FLUX.2 Klein 4B",
        "description": "Fast, Apache 2.0 license, ~13GB VRAM, sub-second on RTX 3090/4070",
        "default_steps": 4,
        "max_steps": 8,
        "pipeline_class": "Flux2KleinPipeline",
        "vram": "~13GB",
        "distilled": True,
        "default_guidance": 1.0
    },
    "klein-4b-fp8": {
        "id": "black-forest-labs/FLUX.2-klein-base-4b-fp8",
        "name": "FLUX.2 Klein 4B FP8",
        "description": "Quantized, faster, ~8GB VRAM, Apache 2.0",
        "default_steps": 4,
        "max_steps": 8,
        "pipeline_class": "Flux2KleinPipeline",
        "vram": "~8GB",
        "distilled": True,
        "default_guidance": 1.0
    },
    "klein-9b": {
        "id": "black-forest-labs/FLUX.2-klein-9B",
        "name": "FLUX.2 Klein 9B",
        "description": "Higher quality, non-commercial license",
        "default_steps": 4,
        "max_steps": 8,
        "pipeline_class": "Flux2KleinPipeline",
        "vram": "~20GB",
        "distilled": True,
        "default_guidance": 1.0
    },
    "dev": {
        "id": "black-forest-labs/FLUX.2-dev",
        "name": "FLUX.2 Dev",
        "description": "32B params, highest quality, requires H100-level GPU",
        "default_steps": 28,
        "max_steps": 50,
        "pipeline_class": "Flux2Pipeline",
        "vram": "~80GB",
        "distilled": False,
        "default_guidance": 3.5
    }
}

# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_key: str):
    """Load a FLUX.2 model into memory."""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    if model_state.current_model == model_key and model_state.pipeline is not None:
        logger.info(f"Model {model_key} already loaded")
        return
    
    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["id"]
    pipeline_class_name = model_info["pipeline_class"]
    
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Device: {model_state.device}, dtype: {model_state.dtype}")
    
    # Clear previous model from memory
    if model_state.pipeline is not None:
        del model_state.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        # Import the appropriate pipeline class
        from diffusers import Flux2KleinPipeline, Flux2Pipeline
        
        if pipeline_class_name == "Flux2KleinPipeline":
            PipelineClass = Flux2KleinPipeline
        else:
            PipelineClass = Flux2Pipeline
        
        # Load pipeline
        model_state.pipeline = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=model_state.dtype,
        )
        
        # Enable memory optimizations
        model_state.pipeline.enable_model_cpu_offload()
        
        model_state.current_model = model_key
        logger.info(f"Model {model_key} loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# =============================================================================
# Pydantic Models
# =============================================================================

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None  # None = use model default (1.0 for distilled, 3.5 for dev)
    seed: Optional[int] = None
    num_images: Optional[int] = 1  # Number of images to generate (1-6)

class Img2ImgRequest(BaseModel):
    prompt: str
    strength: Optional[float] = 0.75
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None  # None = use model default (1.0 for distilled, 3.5 for dev)
    seed: Optional[int] = None

class ModelSwitchRequest(BaseModel):
    model: str

class GenerateResponse(BaseModel):
    success: bool
    image: Optional[str] = None  # Base64 encoded image (single image, backward compatible)
    images: Optional[list] = None  # List of {image: base64, seed: int} for multiple images
    error: Optional[str] = None
    seed: Optional[int] = None

# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load default model on startup."""
    logger.info("Starting FLUX.2 server...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"VRAM: {vram_gb:.1f} GB")
        
        # Choose default model based on available VRAM
        if vram_gb >= 80:
            default_model = "dev"
        elif vram_gb >= 12:
            default_model = "klein-4b"
        else:
            default_model = "klein-4b"  # Will use CPU offload for smaller VRAM
    else:
        default_model = "klein-4b"
    
    # Load default model
    try:
        load_model(default_model)
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down FLUX.2 server...")

app = FastAPI(
    title="FLUX.2 Image Generation API",
    description="Local FLUX.2 image generation server with text-to-image and image editing",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "current_model": model_state.current_model,
        "device": model_state.device
    }

@app.get("/api/models")
async def list_models():
    """List available FLUX.2 models."""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "default_steps": info["default_steps"],
            "vram": info["vram"],
            "loaded": model_state.current_model == key
        })
    return {"models": models, "current": model_state.current_model}

@app.post("/api/load-model")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different FLUX.2 model."""
    try:
        load_model(request.model)
        return {"success": True, "model": request.model}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate one or more images from a text prompt using FLUX.2."""
    if model_state.pipeline is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Determine number of steps and guidance
        model_info = AVAILABLE_MODELS[model_state.current_model]
        num_steps = request.num_inference_steps or model_info["default_steps"]
        num_steps = min(num_steps, model_info["max_steps"])
        guidance = request.guidance_scale if request.guidance_scale is not None else model_info["default_guidance"]
        
        # Clamp num_images to 1-6
        num_images = max(1, min(6, request.num_images or 1))
        
        # Set base seed for reproducibility
        base_seed = request.seed
        if base_seed is None:
            base_seed = torch.randint(0, 2**32, (1,)).item()
        
        logger.info(f"Generating {num_images} image(s): prompt='{request.prompt[:50]}...', steps={num_steps}, guidance={guidance}, base_seed={base_seed}")
        
        generated_images = []
        
        for i in range(num_images):
            # Each image gets a unique seed (base_seed + i)
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            # Generate image
            result = model_state.pipeline(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            )
            
            image = result.images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            generated_images.append({
                "image": image_base64,
                "seed": current_seed
            })
            
            logger.info(f"Image {i+1}/{num_images} generated with seed {current_seed}")
        
        logger.info(f"All {num_images} image(s) generated successfully")
        
        # Return response - include both single image (backward compatible) and images array
        return GenerateResponse(
            success=True,
            image=generated_images[0]["image"] if generated_images else None,
            images=generated_images,
            seed=generated_images[0]["seed"] if generated_images else None
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerateResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/img2img", response_model=GenerateResponse)
async def image_to_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.75),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: float = Form(4.0),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1)
):
    """Edit an image using FLUX.2 with a text prompt. Supports multiple outputs."""
    if model_state.pipeline is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Load the input image
        image_data = await image.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize to supported dimensions if needed
        width, height = init_image.size
        # Round to nearest 64 for model compatibility
        width = (width // 64) * 64
        height = (height // 64) * 64
        width = max(512, min(width, 1280))
        height = max(512, min(height, 1280))
        if width != init_image.width or height != init_image.height:
            init_image = init_image.resize((width, height), Image.LANCZOS)
        
        # Determine number of steps and guidance
        model_info = AVAILABLE_MODELS[model_state.current_model]
        num_steps = num_inference_steps or model_info["default_steps"]
        num_steps = min(num_steps, model_info["max_steps"])
        guidance = guidance_scale if guidance_scale is not None else model_info["default_guidance"]
        
        # Clamp num_images to 1-6
        num_images = max(1, min(6, num_images))
        
        # Set base seed
        base_seed = seed
        if base_seed is None:
            base_seed = torch.randint(0, 2**32, (1,)).item()
        
        logger.info(f"Img2Img: {num_images} image(s), prompt='{prompt[:50]}...', steps={num_steps}, guidance={guidance}")
        
        generated_images = []
        
        for i in range(num_images):
            # Each image gets a unique seed
            current_seed = base_seed + i
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            # Generate image (FLUX.2 Klein uses image as reference, doesn't use strength)
            result = model_state.pipeline(
                prompt=prompt,
                image=init_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            )
            
            output_image = result.images[0]
            
            # Convert to base64
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            generated_images.append({
                "image": image_base64,
                "seed": current_seed
            })
            
            logger.info(f"Img2Img {i+1}/{num_images} completed with seed {current_seed}")
        
        logger.info(f"All {num_images} Img2Img image(s) completed successfully")
        
        return GenerateResponse(
            success=True,
            image=generated_images[0]["image"] if generated_images else None,
            images=generated_images,
            seed=generated_images[0]["seed"] if generated_images else None
        )
        
    except Exception as e:
        logger.error(f"Img2Img failed: {e}")
        return GenerateResponse(
            success=False,
            error=str(e)
        )

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
