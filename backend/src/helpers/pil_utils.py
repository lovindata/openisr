from PIL.Image import Image


def build_thumbnail(image: Image, square_length: int) -> Image:
    def center_square(image: Image) -> Image:
        origin_width, origin_height = image.size
        if origin_width > origin_height:
            border_to_crop_pixels = (origin_width - origin_height) // 2
            image = image.crop(
                (
                    border_to_crop_pixels,
                    0,
                    origin_width - border_to_crop_pixels,
                    origin_height,
                )
            )
        else:
            border_to_crop_pixels = (origin_height - origin_width) // 2
            image = image.crop(
                (
                    0,
                    border_to_crop_pixels,
                    origin_width,
                    origin_height - border_to_crop_pixels,
                )
            )
        square_length = min(image.size)
        return image.resize((square_length, square_length))

    image = center_square(image)
    image.thumbnail((square_length, square_length))
    return image
