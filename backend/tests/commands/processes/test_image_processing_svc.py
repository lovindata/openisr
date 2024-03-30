from io import BytesIO
from itertools import product
from unittest.mock import Mock

import pytest
from PIL.Image import Image, new, open

from backend.v2.commands.processes.services.image_processing_svc import (
    image_processing_svc_impl,
)
from backend.v2.commands.shared.models.extension_val import ExtensionVal


class TestImageProcessingSvc:
    @pytest.mark.parametrize(
        "input_extension, input_enable_ai, output_extension",
        list(product(list(ExtensionVal), [False, True], list(ExtensionVal))),
    )
    def test_process_image_mutually_convertible_extensions(
        self,
        input_extension: ExtensionVal,
        input_enable_ai: bool,
        output_extension: ExtensionVal,
    ) -> None:
        def build_input_image() -> Image:
            input_image = new("RGBA", (4, 4))
            bytesio = BytesIO()
            if input_extension == ExtensionVal.JPEG:
                input_image = input_image.convert("RGB")
            input_image.save(bytesio, format=input_extension.value)
            input_image = open(bytesio)
            return input_image

        input_image = build_input_image()
        input_process = Mock(
            enable_ai=input_enable_ai,
            extension=output_extension,
            target=Mock(width=input_image.size[0] * 2, height=input_image.size[1] * 2),
        )
        output_image = image_processing_svc_impl.process_image(
            input_image, input_process
        )
        assert ExtensionVal(output_image.format) == output_extension
