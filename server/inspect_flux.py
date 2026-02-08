
try:
    from diffusers import Flux2KleinPipeline
    import inspect
    print("Flux2KleinPipeline Signature Parameters:")
    sig = inspect.signature(Flux2KleinPipeline.__call__)
    for name, param in sig.parameters.items():
        print(f"{name}")
except ImportError:
    print("Flux2KleinPipeline not found.")
    try:
        from diffusers import FluxPipeline
        import inspect
        print("FluxPipeline Signature Parameters:")
        sig = inspect.signature(FluxPipeline.__call__)
        for name, param in sig.parameters.items():
            print(f"{name}")
    except:
        pass
except Exception as e:
    print(f"Error: {e}")
