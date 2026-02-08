
try:
    from diffusers import Flux2KleinPipeline
    
    if hasattr(Flux2KleinPipeline, 'load_ip_adapter'):
        print("Flux2KleinPipeline has load_ip_adapter method.")
    else:
        print("Flux2KleinPipeline does NOT have load_ip_adapter method.")
        
except ImportError:
    print("Flux2KleinPipeline not found.")
except Exception as e:
    print(f"Error: {e}")
