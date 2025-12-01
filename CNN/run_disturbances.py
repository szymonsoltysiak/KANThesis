### RUN_DISTURBANCES.PY
# Script to apply various disturbances to images from the GTSRB dataset.
# Usage example:
# python run_disturbances.py --index 10 --blur --blur_radius 3 --rain --rain_percentage 0.3 --brightness --brightness_factor 1.2 --rotation --rotation_angle 15 --show both
# Flags:
# --index: Choose test image by index (if omitted: random image)
# --blur: Apply blur effect
# --blur_radius: Blur intensity (default 2.0)
# --rain: Apply simulated rain
# --rain_percentage: Rain density as float 0.0–1.0 (default 0.2)
# --brightness: Adjust brightness
# --brightness_factor: Brightness factor (>1 brighter, <1 darker)
# --rotation: Rotate image
# --rotation_angle: Rotation angle in degrees
# --show: What to show: original, disturbed, or both (default)

import argparse
import random
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

from disturbances import (
    apply_blur,
    add_rain_effect,
    apply_brightness,
    apply_rotation
)

def load_image_by_index(index=None):
    dataset = datasets.GTSRB(
        root="./data",
        split="test",
        transform=None,
        download=True
    )

    if index is None:
        index = random.randint(0, len(dataset) - 1)

    if index < 0 or index >= len(dataset):
        raise ValueError(f"Index {index} out of range! Dataset size: {len(dataset)}")

    image, label = dataset[index]

    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)

    return image, label, index


def apply_selected_disturbances(
    image,
    use_blur=False,
    blur_radius=2,
    use_rain=False,
    rain_percentage=0.2,
    use_brightness=False,
    brightness_factor=1.5,
    use_rotation=False,
    rotation_angle=20
):
    disturbed = image

    if use_rotation:
        disturbed = apply_rotation(disturbed, angle=rotation_angle)

    if use_blur:
        disturbed = apply_blur(disturbed, blur_radius=blur_radius)

    if use_brightness:
        disturbed = apply_brightness(disturbed, brightness_factor=brightness_factor)

    if use_rain:
        disturbed = add_rain_effect(disturbed, rain_percentage=rain_percentage)

    return disturbed


def main():
    parser = argparse.ArgumentParser(description="Apply disturbances to a GTSRB image.")

    # Choosing image
    parser.add_argument("--index", type=int, default=None,
                        help="Choose test image by index (if omitted: random image)")

    # Disturbances
    parser.add_argument("--blur", action="store_true", help="Apply blur effect")
    parser.add_argument("--blur_radius", type=float, default=2.0,
                        help="Blur intensity (default 2.0)")

    parser.add_argument("--rain", action="store_true", help="Apply simulated rain")
    parser.add_argument("--rain_percentage", type=float, default=0.2,
                        help="Rain density as float 0.0–1.0 (default 0.2)")

    parser.add_argument("--brightness", action="store_true", help="Adjust brightness")
    parser.add_argument("--brightness_factor", type=float, default=1.5,
                        help="Brightness factor (>1 brighter, <1 darker)")

    parser.add_argument("--rotation", action="store_true", help="Rotate image")
    parser.add_argument("--rotation_angle", type=float, default=20,
                        help="Rotation angle in degrees")

    # Show mode
    parser.add_argument("--show", type=str, default="both",
                        choices=["both", "disturbed", "original"],
                        help="What to show: original, disturbed, or both (default)")

    args = parser.parse_args()

    image, label, used_index = load_image_by_index(args.index)

    disturbed = apply_selected_disturbances(
        image,
        use_blur=args.blur,
        blur_radius=args.blur_radius,
        use_rain=args.rain,
        rain_percentage=args.rain_percentage,
        use_brightness=args.brightness,
        brightness_factor=args.brightness_factor,
        use_rotation=args.rotation,
        rotation_angle=args.rotation_angle
    )

    # DISPLAY LOGIC
    if args.show == "both":
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Original (index = {used_index})")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(disturbed)
        plt.title("Disturbed")
        plt.axis("off")

    elif args.show == "original":
        plt.imshow(image)
        plt.title(f"Original (index = {used_index})")
        plt.axis("off")

    elif args.show == "disturbed":
        plt.imshow(disturbed)
        plt.title("Disturbed Image")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()        
