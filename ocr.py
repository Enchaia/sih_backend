from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from paddleocr import PaddleOCR


PlateImage = Union[str, Path, np.ndarray]


# Initialize once and reuse
ocr_engine = PaddleOCR(
    lang="en",
    # Avoid the PaddlePaddle 3.x CPU oneDNN/MKLDNN compatibility failure.
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def _clean_plate_text(text: str) -> str:
    """Convert text such as 'DL 01 AB 1234' -> 'DL01AB1234'."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _load_image(plate_image: PlateImage) -> np.ndarray:

    if isinstance(plate_image, (str, Path)):
        image = cv2.imread(str(plate_image))

        if image is None:
            raise ValueError(
                f"Could not read plate image: {plate_image}"
            )

        return image

    if isinstance(plate_image, np.ndarray):

        if plate_image.size == 0:
            raise ValueError("Plate image is empty.")

        return plate_image

    raise TypeError(
        "plate_image must be an image path or NumPy/OpenCV image."
    )


def recognize_plate(plate_image: PlateImage) -> dict:

    image = _load_image(plate_image)

    results = ocr_engine.predict(image)

    texts = []
    confidences = []

    for result in results:

        # PaddleOCR 3.x result object
        data = result.json

        if callable(data):
            data = data()

        if not data:
            continue

        if isinstance(data, dict):
            data = data.get("res", data)

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        for text, score in zip(rec_texts, rec_scores):

            cleaned = _clean_plate_text(str(text))

            if not cleaned:
                continue

            texts.append(cleaned)
            confidences.append(float(score))

    if not texts:
        return {
            "text": "",
            "confidence": 0.0
        }

    plate_text = "".join(texts)

    confidence = sum(confidences) / len(confidences)

    return {
        "text": plate_text,
        "confidence": round(confidence, 4)
    }


if __name__ == "__main__":
    # Change this path when testing another cropped plate image.
    sample_image = Path(__file__).with_name("test.png")
    result = recognize_plate(sample_image)
    print(result)
