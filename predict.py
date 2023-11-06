# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

MODEL_NAME = "RealESRGAN_x4plus"
ESRGAN_PATH = os.path.join("/root/.cache/realesrgan", MODEL_NAME + ".pth")
GFPGAN_PATH = "/root/.cache/realesrgan/GFPGANv1.3.pth"


def crop_to_exact_size(image, target_width, target_height):
    # Calculate the margins to crop from the center of the image
    margin_x = max((image.width - target_width) // 2, 0)
    margin_y = max((image.height - target_height) // 2, 0)

    # Define the crop area
    crop_area = (
        margin_x,
        margin_y,
        margin_x + target_width,
        margin_y + target_height
    )

    # Crop the image to the calculated area
    cropped_image = image.crop(crop_area)
    return cropped_image


class Predictor(BasePredictor):
    def setup(self):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=ESRGAN_PATH,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.face_enhancer = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def predict(
            self,
            image: Path = Input(description="Input image"),
            scale: float = Input(
                description="Factor to scale image by", ge=0, le=15, default=4
            ),
            width: int = Input(
                description="Desired width of the output image", default=None
            ),
            height: int = Input(
                description="Desired height of the output image", default=None
            ),
            face_enhance: bool = Input(
                description="Run GFPGAN face enhancement along with upscaling",
                default=False,
            ),
    ) -> Path:
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)

        if face_enhance:
            print("running with face enhancement")
            self.face_enhancer.upscale = scale
            _, _, output = self.face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            print("running without face enhancement")
            output, _ = self.upsampler.enhance(img, outscale=scale)

        # Resize the image to the specified width and height if they are provided
        if width is not None and height is not None:
            if not isinstance(output, Image.Image):
                if output.dtype != np.uint8:
                    output = output.astype(np.uint8)
                output = Image.fromarray(output)
            # Resize the image to the original size before saving
            output = crop_to_exact_size(output, width, height)

        save_path = os.path.join(tempfile.mkdtemp(), "output.png")
        output.save(save_path)
        return Path(save_path)
