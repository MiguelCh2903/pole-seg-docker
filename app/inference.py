"""YOLO Segmentation inference engine using Ultralytics."""

import logging
from typing import Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO

from .config import config

logger = logging.getLogger(__name__)


class YOLOSegmentationInference:
    """YOLOv8 Segmentation inference engine using Ultralytics library."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        input_size: int = 640,
        use_gpu: bool = False,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to ONNX or TensorRT model
            confidence_threshold: Minimum confidence for detections
            input_size: Model input size
            use_gpu: Whether to use GPU (TensorRT) or CPU (ONNX)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.use_gpu = use_gpu

        # Determine device based on model type and GPU availability
        if self.model_path.endswith(".engine"):
            self.device = 0  # GPU for TensorRT
            self.model_type = "TensorRT"
            logger.info("Loading TensorRT model for GPU inference")
        elif self.model_path.endswith(".onnx"):
            self.device = "cpu"
            self.model_type = "ONNX"
            logger.info("Loading ONNX model for CPU inference")
        else:
            raise ValueError(
                "Model must be either .onnx (CPU) or .engine (TensorRT GPU)"
            )

        # Load YOLO model
        logger.info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path, task="segment")

        logger.info(f"Model loaded successfully")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Device: {self.device}")

        # Get class names if available
        try:
            if hasattr(self.model, "names"):
                logger.info(f"Class names: {self.model.names}")
        except Exception as e:
            logger.warning(f"Could not retrieve class names: {e}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an image.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Dictionary containing:
                - boxes: Detection boxes [x1, y1, x2, y2]
                - scores: Confidence scores
                - class_ids: Class IDs
                - masks: Segmentation masks
                - num_detections: Number of detections
                - original_shape: Original image shape
        """
        # Get original image shape
        original_shape = image.shape[:2]

        # Run inference using ultralytics
        logger.debug(f"Running inference on image shape: {image.shape}")

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            save=False,
            verbose=False,
            imgsz=self.input_size,
        )

        # Process results
        result = results[0]

        num_detections = len(result.boxes)
        logger.info(f"Found {num_detections} detections")

        if num_detections == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "class_ids": np.array([]),
                "masks": [],
                "num_detections": 0,
                "original_shape": original_shape,
            }

        # Extract boxes, scores, and classes
        boxes = []
        scores = []
        class_ids = []
        masks = []

        for i, box in enumerate(result.boxes):
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Get bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy()

            boxes.append(xyxy)
            scores.append(conf)
            class_ids.append(cls)

        # Extract masks if available
        if result.masks is not None:
            for mask_data in result.masks.data:
                # Convert mask to numpy array
                mask = mask_data.cpu().numpy()

                # Resize mask to original image size if needed
                if mask.shape != original_shape:
                    mask = cv2.resize(
                        mask,
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                masks.append(mask)

        return {
            "boxes": np.array(boxes),
            "scores": np.array(scores),
            "class_ids": np.array(class_ids),
            "masks": masks,
            "num_detections": num_detections,
            "original_shape": original_shape,
            "result": result,  # Keep the original result for visualization
        }

    def predict_and_visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference and return annotated image.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Annotated image with detections visualized
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            save=False,
            verbose=False,
            imgsz=self.input_size,
        )

        result = results[0]
        annotated_image = result.plot()

        return annotated_image


# Global inference engine instance
_inference_engine = None


def get_inference_engine() -> YOLOSegmentationInference:
    """Get or create the global inference engine instance."""
    global _inference_engine

    if _inference_engine is None:
        _inference_engine = YOLOSegmentationInference(
            model_path=config.MODEL_PATH,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            input_size=config.MODEL_INPUT_SIZE,
            use_gpu=config.USE_GPU,
        )

    return _inference_engine
