# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import tempfile
from typing import Optional

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path
from gfpgan import GFPGANer
from realesrgan import RealESRGANer

MODEL_NAME = "RealESRGAN_x4plus"
ESRGAN_PATH = os.path.join("/root/.cache/realesrgan", MODEL_NAME + ".pth")
GFPGAN_PATH = "/root/.cache/realesrgan/GFPGANv1.3.pth"


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
            width: Optional[int] = Input(
                description="Desired width of the output image", default=None
            ),
            height: Optional[int] = Input(
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
            output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)

        save_path = os.path.join(tempfile.mkdtemp(), "output.png")
        cv2.imwrite(save_path, output)
        return Path(save_path)
