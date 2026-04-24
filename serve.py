from waitress import serve
from app import app, load_model
import os

if __name__ == "__main__":
    # Ensure model is loaded before starting the server
    print("Initializing CAN+CBAM Model...")
    load_model(use_segmentation=True)
    
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting production server on http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop.")
    
    # threads=4 is a good default for PyTorch inference
    serve(app, host='0.0.0.0', port=port, threads=4)
