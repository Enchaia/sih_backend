"""
Pre-API frame quality check.

One check runs on every extracted frame before it's sent to the detection
API:

- Lens-health check — is this frame too blurred/washed out to trust
  (motion blur, glare, rain-smeared or dirty lens)? Flagged, not skipped —
  it still gets processed, but tagged so a persistent run of degraded
  frames can raise a camera-health alert instead of being silently read as
  "no hazards here".
"""

import cv2
from config import BLUR_VARIANCE_THRESHOLD


def lens_health(frame, threshold=BLUR_VARIANCE_THRESHOLD):
    """Returns (is_degraded, blur_score) via Laplacian variance.

    Low variance = flat/blurry image: motion blur, glare wash-out, or a
    dirty/wet/obstructed lens. High variance = sharp edges present.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_degraded = bool(variance < threshold)  # cast away numpy.bool_ -> native Python bool
    return is_degraded, round(float(variance), 2)
 

