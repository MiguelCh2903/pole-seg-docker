"""Image processing utilities."""

import cv2
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def preprocess_image(
    image: np.ndarray, target_size: int = 640
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
    """
    Preprocess image for YOLO inference.

    Args:
        image: Input image (BGR format)
        target_size: Target size for the model

    Returns:
        Tuple of (preprocessed_image, original_shape, scale_factors)
    """
    original_height, original_width = image.shape[:2]

    # Calculate scaling to maintain aspect ratio
    scale = target_size / max(original_height, original_width)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create padded image (letterbox)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # Calculate padding
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2

    # Place resized image in center
    padded[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized

    # Convert to RGB and normalize
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32) / 255.0

    # Transpose to CHW format (channels first)
    padded = np.transpose(padded, (2, 0, 1))

    # Add batch dimension
    padded = np.expand_dims(padded, axis=0)

    return padded, (original_height, original_width), (scale, pad_x, pad_y)


def postprocess_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    original_shape: Tuple[int, int],
    input_shape: Tuple[int, int],
    scale_info: Tuple[float, int, int],
) -> List[np.ndarray]:
    """
    Postprocess segmentation masks to original image size.

    Args:
        masks: Predicted masks from model
        boxes: Detection boxes
        original_shape: Original image shape (H, W)
        input_shape: Model input shape
        scale_info: Scaling information (scale, pad_x, pad_y)

    Returns:
        List of binary masks in original image size
    """
    scale, pad_x, pad_y = scale_info
    orig_h, orig_w = original_shape
    processed_masks = []

    for mask in masks:
        # Resize mask to input size
        if mask.shape != input_shape:
            mask = cv2.resize(mask, input_shape, interpolation=cv2.INTER_LINEAR)

        # Remove padding
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        mask = mask[pad_y : pad_y + new_h, pad_x : pad_x + new_w]

        # Resize to original size
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold to binary mask
        mask = (mask > 0.5).astype(np.uint8)

        processed_masks.append(mask)

    return processed_masks


def apply_masks_to_image(
    image: np.ndarray,
    masks: List[np.ndarray],
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Apply segmentation masks to image with colored overlays.

    Args:
        image: Original image (BGR)
        masks: List of binary masks
        boxes: Detection boxes
        scores: Confidence scores
        class_ids: Class IDs
        alpha: Transparency factor for masks

    Returns:
        Image with masks applied
    """
    result = image.copy()

    # Generate distinct colors for each instance
    np.random.seed(42)  # For reproducible colors
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

    for idx, (mask, box, score, class_id) in enumerate(
        zip(masks, boxes, scores, class_ids)
    ):
        color = colors[idx].tolist()

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = color

        # Blend with original image
        result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)

        # Draw bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = f"ID:{idx} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        cv2.putText(
            result,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return result


def create_individual_mask_images(
    original_shape: Tuple[int, int], masks: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Create individual binary mask images.

    Args:
        original_shape: Original image shape (H, W)
        masks: List of binary masks

    Returns:
        List of binary mask images (0-255)
    """
    mask_images = []
    for mask in masks:
        mask_img = (mask * 255).astype(np.uint8)
        mask_images.append(mask_img)

    return mask_images


def encode_image_to_bytes(image: np.ndarray, format: str = ".jpg") -> bytes:
    """
    Encode image to bytes.

    Args:
        image: Image array
        format: Output format (.jpg, .png)

    Returns:
        Encoded image bytes
    """
    success, encoded = cv2.imencode(format, image)
    if not success:
        raise ValueError(f"Failed to encode image to {format}")

    return encoded.tobytes()
