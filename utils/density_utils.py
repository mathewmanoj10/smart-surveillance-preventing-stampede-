import time
import numpy as np


# ============================================================
# CROWD DENSITY
# ============================================================

def calculate_density(
    person_count,
    frame_width,
    frame_height
):
    """
    Calculate crowd density as people per megapixel.

    Input validation:
        person_count >= 0
        frame_width > 0
        frame_height > 0

    Invalid inputs return 0.0.
    """

    # ========================================================
    # INPUT CONVERSION
    # ========================================================

    try:

        person_count = float(
            person_count
        )

        frame_width = float(
            frame_width
        )

        frame_height = float(
            frame_height
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    # ========================================================
    # FINITE NUMBER CHECK
    # ========================================================

    if not np.isfinite(
        person_count
    ):

        return 0.0

    if not np.isfinite(
        frame_width
    ):

        return 0.0

    if not np.isfinite(
        frame_height
    ):

        return 0.0

    # ========================================================
    # RANGE VALIDATION
    # ========================================================

    if person_count < 0:

        return 0.0

    if frame_width <= 0:

        return 0.0

    if frame_height <= 0:

        return 0.0

    # ========================================================
    # FRAME AREA
    # ========================================================

    frame_area = (

        frame_width
        * frame_height

    )

    if frame_area <= 0:

        return 0.0

    if not np.isfinite(
        frame_area
    ):

        return 0.0

    # ========================================================
    # PEOPLE PER MEGAPIXEL
    # ========================================================

    density = (

        person_count
        / frame_area

    ) * 1_000_000

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if not np.isfinite(
        density
    ):

        return 0.0

    return round(
        float(density),
        4
    )


# ============================================================
# FPS COUNTER
# ============================================================

class FPSCounter:

    def __init__(self):

        self.prev_time = time.time()

        self.fps = 0.0


    # ========================================================
    # UPDATE FPS
    # ========================================================

    def update(self):
        """
        Calculate the current processing FPS.

        Returns:
            float: current FPS
        """

        current_time = time.time()

        elapsed = (

            current_time
            - self.prev_time

        )

        self.prev_time = current_time

        # ====================================================
        # VALIDATE ELAPSED TIME
        # ====================================================

        if not np.isfinite(
            elapsed
        ):

            self.fps = 0.0

            return 0.0

        if elapsed <= 0:

            self.fps = 0.0

            return 0.0

        # ====================================================
        # CALCULATE FPS
        # ====================================================

        calculated_fps = (

            1.0
            / elapsed

        )

        # ====================================================
        # VALIDATE FPS
        # ====================================================

        if not np.isfinite(
            calculated_fps
        ):

            self.fps = 0.0

            return 0.0

        if calculated_fps < 0:

            self.fps = 0.0

            return 0.0

        self.fps = calculated_fps

        return round(
            self.fps,
            2
        )


    # ========================================================
    # GET CURRENT FPS
    # ========================================================

    def get_fps(self):

        if not np.isfinite(
            self.fps
        ):

            return 0.0

        if self.fps < 0:

            return 0.0

        return self.fps


# ============================================================
# CIRCULAR DIRECTION VARIANCE
# ============================================================

def calculate_circular_variance(
    angles
):
    """
    Calculate circular variance of movement directions.

    Range:
        0.0 -> highly consistent direction
        1.0 -> highly varied direction

    Invalid inputs return 0.0.
    """

    # ========================================================
    # NONE CHECK
    # ========================================================

    if angles is None:

        return 0.0

    # ========================================================
    # LENGTH CHECK
    # ========================================================

    try:

        if len(angles) == 0:

            return 0.0

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    # ========================================================
    # CONVERT TO NUMPY ARRAY
    # ========================================================

    try:

        angles = np.asarray(

            angles,

            dtype=float

        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    # ========================================================
    # ARRAY VALIDATION
    # ========================================================

    if angles.size == 0:

        return 0.0

    if not np.all(
        np.isfinite(
            angles
        )
    ):

        return 0.0

    # ========================================================
    # FLATTEN ARRAY
    # ========================================================

    angles = angles.reshape(-1)

    # ========================================================
    # MEAN SINE
    # ========================================================

    mean_sin = np.mean(
        np.sin(angles)
    )

    # ========================================================
    # MEAN COSINE
    # ========================================================

    mean_cos = np.mean(
        np.cos(angles)
    )

    # ========================================================
    # VALIDATE MEAN VALUES
    # ========================================================

    if not np.isfinite(
        mean_sin
    ):

        return 0.0

    if not np.isfinite(
        mean_cos
    ):

        return 0.0

    # ========================================================
    # RESULTANT VECTOR
    # ========================================================

    resultant_length = np.sqrt(

        mean_sin ** 2
        + mean_cos ** 2

    )

    # ========================================================
    # VALIDATE RESULTANT
    # ========================================================

    if not np.isfinite(
        resultant_length
    ):

        return 0.0

    # ========================================================
    # CIRCULAR VARIANCE
    # ========================================================

    variance = (

        1.0
        - resultant_length

    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if not np.isfinite(
        variance
    ):

        return 0.0

    # ========================================================
    # CLAMP RANGE
    # ========================================================

    variance = max(
        0.0,
        min(
            float(variance),
            1.0
        )
    )

    return round(
        variance,
        4
    )
