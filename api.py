import os
import torch
import pickle
import gdown
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# Import your model class
from model_definition import MultimodalEmbeddingModel

# =========================================================================================
# --- 1. API Setup ---
# =========================================================================================

app = FastAPI(
    title="Multimodal Biometric Authentication API",
    description="API to generate user embeddings from multimodal sensor data.",
    version="1.0.0"
)

# Device and model config
DEVICE = torch.device("cpu")
MAX_LEN = 1000
MODEL_PARAMS = {
    'hidden_dim': 128,
    'proj_dim': 128,
    'tcn_layers': 5,
    'dropout_rate': 0.4,
    'sequence_length': MAX_LEN
}

SENSOR_LIST = [
    'key_data', 'swipe', 'touch_touch', 'sensor_grav', 'sensor_gyro',
    'sensor_lacc', 'sensor_magn', 'sensor_nacc'
]
SENSOR_DIMS = {
    'key_data': 1, 'swipe': 6, 'touch_touch': 6, 'sensor_grav': 3,
    'sensor_gyro': 3, 'sensor_lacc': 3, 'sensor_magn': 3, 'sensor_nacc': 3
}
MAX_FEATURE_DIM = max(SENSOR_DIMS.values())

# =========================================================================================
# --- 2. Load the Model (from Google Drive if needed) ---
# =========================================================================================

MODEL_FILE = "multimodal_authentication_model.pkl"
GDRIVE_FILE_ID = "1KiJ174F4NEZAFkROkV3ahMC73pYJfQII"

# Download if not present
if not os.path.exists(MODEL_FILE):
    print("[INFO] Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)
    print("[INFO] Model download complete.")

# Load model
model = MultimodalEmbeddingModel(SENSOR_LIST, SENSOR_DIMS, MODEL_PARAMS)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    # Initialize with random weights if model file is corrupted
    print("[WARNING] Using model with random weights.")

model.eval()
model.to(DEVICE)
print("[INFO] Model ready for inference.")

# =========================================================================================
# --- 3. Input/Output Schemas ---
# =========================================================================================

class SensorData(BaseModel):
    data: Dict[str, List[List[float]]] = Field(default_factory=dict)

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    message: str

# =========================================================================================
# --- 4. Preprocessing Logic ---
# =========================================================================================

def preprocess_input(sensor_data: SensorData) -> torch.Tensor:
    """Preprocess sensor data into model input format."""
    tensors = []

    for sensor in SENSOR_LIST:
        sensor_readings = sensor_data.data.get(sensor, [])
        if not sensor_readings:
            # Create zero tensor for missing sensor data
            data = torch.zeros(MAX_LEN, SENSOR_DIMS[sensor])
        else:
            try:
                df = pd.DataFrame(sensor_readings)
                if not df.empty and df.std().sum() > 0:
                    # Normalize the data
                    df = (df - df.mean()) / (df.std().replace(0, 1))
                df.fillna(0, inplace=True)
                data = torch.tensor(df.values, dtype=torch.float32)
            except Exception as e:
                print(f"[WARNING] Error processing {sensor} data: {e}")
                data = torch.zeros(MAX_LEN, SENSOR_DIMS[sensor])

        # Pad or truncate time steps
        T, D = data.shape
        if T > MAX_LEN:
            data = data[:MAX_LEN]
        elif T < MAX_LEN:
            padding = torch.zeros(MAX_LEN - T, D)
            data = torch.cat([data, padding], dim=0)

        tensors.append(data)

    # Pad feature dimensions to max dimension
    padded = []
    for t in tensors:
        if t.shape[1] < MAX_FEATURE_DIM:
            pad = torch.zeros(t.shape[0], MAX_FEATURE_DIM - t.shape[1])
            t = torch.cat([t, pad], dim=1)
        padded.append(t)

    # Stack and return: (1, Modalities, Time, Features)
    return torch.stack(padded).unsqueeze(0)

# =========================================================================================
# --- 5. API Endpoints ---
# =========================================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for health check."""
    return HealthResponse(
        status="healthy",
        message="Welcome to the Multimodal Authentication API. Visit /docs for usage."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is running successfully."
    )

@app.post("/predict", response_model=EmbeddingResponse)
async def predict(sensor_data: SensorData):
    """
    Generate user embedding from multimodal sensor data.
    
    Args:
        sensor_data: Dictionary containing sensor readings for different modalities
        
    Returns:
        EmbeddingResponse containing the generated embedding vector
    """
    try:
        # Validate input
        if not sensor_data.data:
            raise HTTPException(
                status_code=400, 
                detail="No sensor data provided. Please include at least one sensor modality."
            )
        
        # Preprocess input
        input_tensor = preprocess_input(sensor_data).to(DEVICE)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(input_tensor)
        
        # Convert to list for JSON serialization
        embedding_list = embedding.cpu().numpy().flatten().tolist()
        
        return EmbeddingResponse(
            embedding=embedding_list,
            status="success"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"[ERROR] Inference error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during inference: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_parameters": MODEL_PARAMS,
        "supported_sensors": SENSOR_LIST,
        "sensor_dimensions": SENSOR_DIMS,
        "max_sequence_length": MAX_LEN,
        "device": str(DEVICE),
        "status": "loaded"
    }

# =========================================================================================
# --- 6. Application Startup ---
# =========================================================================================

if __name__ == "__main__":
    # Get port from environment variable (Render provides this automatically)
    port = int(os.environ.get("PORT", 8000))
    
    print(f"[INFO] Starting server on host 0.0.0.0 and port {port}")
    
    # Run the application
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
