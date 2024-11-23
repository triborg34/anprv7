from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config_manager import initialize_config, save_or_update_config, load_config, add_camera_ip

# Initialize the configuration file
initialize_config()
#uvicorn configServer:app --reload --host 0.0.0.0 --port 8000
# FastAPI app
app = FastAPI()

# Pydantic model for updating configurations
class ConfigUpdateRequest(BaseModel):
    section: str
    key: str
    value: str


class CameraIPRequest(BaseModel):
    ip: str


@app.get("/config")
def get_config():
    """Retrieve the entire configuration."""
    config = load_config()
    return {"status": "success", "config": config}


@app.post("/config")
def update_config(request: ConfigUpdateRequest):
    """Update or add a configuration key-value pair."""
    try:
        # Save or update the configuration
        save_or_update_config(request.section, request.key, request.value)
        return {"status": "success", "message": f"{request.key} updated to {request.value} in section {request.section}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








@app.post("/cameras")
def add_camera(request: CameraIPRequest):
    """Add a new camera IP."""
    try:
        rtspIp=f"rtsp://{request.ip}:554/mainstream"
        new_camera = add_camera_ip(rtspIp)
        return {"status": "success", "new_camera": new_camera}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
