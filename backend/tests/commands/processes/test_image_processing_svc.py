from io import BytesIO
from itertools import product
from unittest.mock import Mock

import pytest
from PIL.Image import Image, new, open

from backend.v2.commands.processes.models.process_mod.process_ai_val import ProcessAIVal
from backend.v2.commands.processes.models.process_mod.process_bicubic_val import (
    ProcessBicubicVal,
)
from backend.v2.commands.processes.models.process_mod.process_resolution_val import (
    ProcessResolutionVal,
)
from backend.v2.commands.processes.services.image_processing_svc import (
    image_processing_svc_impl,
)
from backend.v2.commands.shared.models.extension_val import ExtensionVal


class TestImageProcessingSvc:
    @pytest.mark.parametrize(
        "input_extension, bicubic_or_ai, output_extension",
        list(
            product(
                list(ExtensionVal),
                [
                    ProcessBicubicVal(target=ProcessResolutionVal(width=4, height=4)),
                    ProcessAIVal(scale=2),
                ],
                list(ExtensionVal),
            )
        ),
    )
    def test_process_image_mutually_convertible_extensions(
        self,
        input_extension: ExtensionVal,
        bicubic_or_ai: ProcessBicubicVal | ProcessAIVal,
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
        input_process = Mock(scaling=bicubic_or_ai, extension=output_extension)
        output_image = image_processing_svc_impl.process_image(
            input_image, input_process
        )
        assert ExtensionVal(output_image.format) == output_extension
