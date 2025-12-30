"""Configuration module for the inference service."""

import os
from pathlib import Path


class Config:
    """Application configuration."""

    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/models/yolo_seg_model.onnx")
    MODEL_INPUT_SIZE: int = int(os.getenv("MODEL_INPUT_SIZE", "640"))

    # Inference settings
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", "100"))

    # Device settings - Check if using TensorRT (.engine) or ONNX (.onnx)
    # TensorRT models use GPU, ONNX models use CPU
    USE_GPU: bool = os.getenv("USE_GPU", "auto").lower() == "true"

    @staticmethod
    def is_gpu_model() -> bool:
        """Check if the model is a GPU model (TensorRT)."""
        model_path = Config.MODEL_PATH
        return model_path.endswith(".engine")

    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))

    # Upload settings
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp"}

    # Output settings
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/outputs")
    SAVE_OUTPUTS: bool = os.getenv("SAVE_OUTPUTS", "false").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        model_path = Path(cls.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {cls.MODEL_PATH}")

        # Check if model format is supported
        if not (model_path.suffix == ".onnx" or model_path.suffix == ".engine"):
            raise ValueError(
                f"Model must be either .onnx or .engine format. Got: {model_path.suffix}"
            )

        if cls.CONFIDENCE_THRESHOLD < 0 or cls.CONFIDENCE_THRESHOLD > 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")

        # Create output directory if needed
        if cls.SAVE_OUTPUTS:
            Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


config = Config()
