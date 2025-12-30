"""FastAPI application for YOLOv8 Segmentation inference."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import config
from .inference import get_inference_engine
from .utils import (
    apply_masks_to_image,
    create_individual_mask_images,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="YOLOv8 Segmentation Inference API",
    description="Instance segmentation service using YOLOv8 ONNX model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting YOLOv8 Segmentation Inference Service")

    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Initialize inference engine
    try:
        engine = get_inference_engine()
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "YOLOv8 Segmentation Inference API",
        "version": "1.0.0",
        "status": "running",
        "device": get_inference_engine().device,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_inference_engine()
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": engine.device,
            "model_path": config.MODEL_PATH,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_image: bool = Query(True, description="Return annotated image"),
    return_masks: bool = Query(False, description="Return individual masks"),
    confidence: Optional[float] = Query(
        None, description="Override confidence threshold", ge=0.0, le=1.0
    ),
):
    """
    Perform instance segmentation on an uploaded image.

    Args:
        file: Image file to process
        return_image: Whether to return annotated image
        return_masks: Whether to return individual mask images
        confidence: Optional confidence threshold override

    Returns:
        JSON with detection results and optionally annotated image
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}",
        )

    # Read and decode image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Run inference
    try:
        # Override confidence if provided
        if confidence is not None:
            engine = get_inference_engine()
            original_conf = engine.confidence_threshold
            engine.confidence_threshold = confidence

        engine = get_inference_engine()
        results = engine.predict(image)

        # Restore original confidence
        if confidence is not None:
            engine.confidence_threshold = original_conf

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Prepare response
    response_data = {"num_detections": results["num_detections"], "detections": []}

    # Add detection details
    for idx in range(results["num_detections"]):
        detection = {
            "id": idx,
            "box": results["boxes"][idx].tolist(),
            "confidence": float(results["scores"][idx]),
            "class_id": int(results["class_ids"][idx]),
        }
        response_data["detections"].append(detection)

    # Add annotated image if requested
    if return_image and results["num_detections"] > 0:
        try:
            # Use ultralytics built-in visualization
            annotated = results["result"].plot()

            # Encode to bytes
            _, buffer = cv2.imencode(".jpg", annotated)
            img_bytes = buffer.tobytes()

            # Convert to base64 for JSON response
            import base64

            response_data["annotated_image"] = base64.b64encode(img_bytes).decode(
                "utf-8"
            )

        except Exception as e:
            logger.error(f"Error creating annotated image: {e}")
            response_data["annotated_image_error"] = str(e)

    return JSONResponse(content=response_data)


@app.post("/predict/image")
async def predict_return_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = Query(
        None, description="Override confidence threshold", ge=0.0, le=1.0
    ),
):
    """
    Perform instance segmentation and return annotated image directly.

    Args:
        file: Image file to process
        confidence: Optional confidence threshold override

    Returns:
        Annotated image with segmentation masks
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}",
        )

    # Read and decode image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Run inference
    try:
        # Override confidence if provided
        if confidence is not None:
            engine = get_inference_engine()
            original_conf = engine.confidence_threshold
            engine.confidence_threshold = confidence

        engine = get_inference_engine()
        results = engine.predict(image)

        # Restore original confidence
        if confidence is not None:
            engine.confidence_threshold = original_conf

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Generate annotated image
    if results["num_detections"] > 0:
        # Use ultralytics built-in visualization
        annotated = results["result"].plot()
    else:
        annotated = image

    # Encode and return
    _, buffer = cv2.imencode(".jpg", annotated)

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Num-Detections": str(results["num_detections"])},
    )


@app.post("/predict/masks")
async def predict_return_masks(
    file: UploadFile = File(...),
    mask_id: Optional[int] = Query(None, description="Return specific mask by ID"),
    confidence: Optional[float] = Query(
        None, description="Override confidence threshold", ge=0.0, le=1.0
    ),
):
    """
    Perform instance segmentation and return binary masks.

    Args:
        file: Image file to process
        mask_id: Optional specific mask ID to return
        confidence: Optional confidence threshold override

    Returns:
        Binary mask images
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}",
        )

    # Read and decode image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Run inference
    try:
        # Override confidence if provided
        if confidence is not None:
            engine = get_inference_engine()
            original_conf = engine.confidence_threshold
            engine.confidence_threshold = confidence

        engine = get_inference_engine()
        results = engine.predict(image)

        # Restore original confidence
        if confidence is not None:
            engine.confidence_threshold = original_conf

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    if results["num_detections"] == 0:
        raise HTTPException(status_code=404, detail="No detections found")

    # Get mask images
    mask_images = create_individual_mask_images(
        results["original_shape"], results["masks"]
    )

    # Return specific mask if requested
    if mask_id is not None:
        if mask_id < 0 or mask_id >= len(mask_images):
            raise HTTPException(
                status_code=404,
                detail=f"Mask ID {mask_id} not found. Available: 0-{len(mask_images) - 1}",
            )

        _, buffer = cv2.imencode(".png", mask_images[mask_id])
        return Response(content=buffer.tobytes(), media_type="image/png")

    # Return all masks as JSON with base64 encoding
    import base64

    masks_data = []
    for idx, mask_img in enumerate(mask_images):
        _, buffer = cv2.imencode(".png", mask_img)
        mask_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        masks_data.append(
            {"id": idx, "mask": mask_b64, "confidence": float(results["scores"][idx])}
        )

    return JSONResponse(content={"num_masks": len(masks_data), "masks": masks_data})


def main():
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level="info",
    )


if __name__ == "__main__":
    main()
