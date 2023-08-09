import PIL
from typing import Iterable

from ram.models import ram, RAM
from ram import inference_ram
from ram import get_transform


def compute_unlikely_tags_center_crop_ensemble(
    image: PIL.Image,
    image_tags: Iterable[str],
    cc_proportions: Iterable[float],
    model: RAM,
    device,
) -> Iterable[str]:
    """
    Finds unlikely tags in a set of tags for an image by running the
    model on center cropped versions of the original image

    Args:
        image: original image
        image_tags: tags of the original image
        cc_proportions: list of border crop proportions
        model: tagging model
        device: device holding the tagging model

    Returns:
        set of unlikely tags
    """

    def center_crop(img, crop_border_proportion):
        assert crop_border_proportion < 0.5
        width, height = img.size
        return img.crop(
            (
                crop_border_proportion * width,
                crop_border_proportion * height,
                width * (1 - crop_border_proportion),
                height * (1 - crop_border_proportion),
            )
        )

    cc_images = [center_crop(image, ccp) for ccp in cc_proportions]

    transform = get_transform(image_size=model.image_size)

    unlikely_tags_set = set()
    for cc_image in cc_images:
        cc_image_input = transform(cc_image).unsqueeze(0).to(device)
        res = inference_ram(cc_image_input, model)
        cc_image_tags = res[0].split(" | ")

        unlikely_tags = [tag for tag in image_tags if tag not in cc_image_tags]
        unlikely_tags_set.update(unlikely_tags)

    return unlikely_tags_set


def compute_unlikely_tags_contrast_ensemble(
    image: PIL.Image,
    image_tags: Iterable[str],
    contrast_factors: Iterable[float],
    model: RAM,
    device,
) -> Iterable[str]:
    """
    Finds unlikely tags in a set of tags for an image by running the
    model on contrast adjusted version of the original image

    Args:
        image: original image
        image_tags: tags of the original image
        contrast_factors: list of contrast factors
        model: tagging model
        device: device holding the tagging model

    Returns:
        set of unlikely tags
    """

    def modify_contrast(img, contrast_factor):
        enhancer = PIL.ImageEnhance.Contrast(img)
        return enhancer.enhance(contrast_factor)

    mc_images = [modify_contrast(image, cf) for cf in contrast_factors]

    transform = get_transform(image_size=model.image_size)

    unlikely_tags_set = set()
    for mc_image in mc_images:
        mc_image_input = transform(mc_image).unsqueeze(0).to(device)
        res = inference_ram(mc_image_input, model)
        mc_image_tags = res[0].split(" | ")

        unlikely_tags = [tag for tag in image_tags if tag not in mc_image_tags]
        unlikely_tags_set.update(unlikely_tags)

    return unlikely_tags_set
