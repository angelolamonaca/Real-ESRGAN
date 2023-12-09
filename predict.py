# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
import requests
import io
from PIL import Image, ImageDraw
import random
import string

MODEL_NAME = "RealESRGAN_x4plus"
ESRGAN_PATH = os.path.join("/root/.cache/realesrgan", MODEL_NAME + ".pth")
GFPGAN_PATH = "/root/.cache/realesrgan/GFPGANv1.3.pth"


def upload_pil_image(pil_image, mime_type):
    # Convert PIL Image to byte array
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # S3 API Endpoints
    S3_UPLOAD_API_ENDPOINT = 'https://4yt4k9lwkd.execute-api.us-east-1.amazonaws.com/default/neuramare-uploads-genvista-client-s3-upload-getPresignedUrl'
    S3_DOWNLOAD_API_ENDPOINT = 'https://z2mfup4u36.execute-api.us-east-1.amazonaws.com/default/neuramare-uploads-genvista-client-s3-download-getPresignedUrl'

    # Step 1: Get the S3 upload URL
    response = requests.get(S3_UPLOAD_API_ENDPOINT)
    s3_upload_pre_signed_url_response = response.json()
    print('Uploading to: ', s3_upload_pre_signed_url_response['uploadURL'])

    # Step 2: Upload the image to S3
    upload_url = s3_upload_pre_signed_url_response['uploadURL']
    headers = {'Content-Type': mime_type}
    requests.put(upload_url, data=img_byte_arr, headers=headers)

    # Step 3: Get the S3 download URL
    response = requests.post(S3_DOWNLOAD_API_ENDPOINT, json={'key': s3_upload_pre_signed_url_response['key']})
    s3_download_pre_signed_url_response = response.json()
    return s3_download_pre_signed_url_response['imageUrl']


def post_to_ai_service(job_id, image_url, callback_url):
    # Payload for the POST request
    payload = {
        'id': job_id,
        'imageUrl': image_url
    }

    # Perform the POST request
    response = requests.post(callback_url, json=payload)

    # Check if the response status is not 200
    if response.status_code != 200:
        # Raise an exception with the status code and response text
        raise Exception(f"NSFW content detected. Try running it again, or try a different prompt.")

    return


def create_fake_image(width, height):
    # Create a blank image with a solid color (e.g., white)
    image = Image.new('RGB', (width, height), color='white')

    # Generate a random string
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

    # Draw the random text on the image
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), random_text, fill='black')

    return image


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
            callback_url: str = Input(
                description="callback url", default=None
            ),
            job_id: str = Input(
                description="job id", default=None
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
                # Assuming the image was in BGR format, convert it to RGB.
                if output.shape[-1] == 3:  # Check if the image has three channels
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output = Image.fromarray(output)
            # Resize the image to the original size before saving
            output = crop_to_exact_size(output, width, height)
            if job_id:
                if not isinstance(output, Image.Image):
                    if output.dtype != np.uint8:
                        output = output.astype(np.uint8)
                    # Assuming the image was in BGR format, convert it to RGB.
                    if output.shape[-1] == 3:  # Check if the image has three channels
                        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                    output = Image.fromarray(output)
                print("Upload image")
                uploaded_image_url = upload_pil_image(output, "image/png")
                print("Send url to ai service for saving")
                post_to_ai_service(job_id, uploaded_image_url)
                print("Define the output path")
                save_path = os.path.join(tempfile.mkdtemp(), "output.png")
                print("Create fake image, since we did the upload")
                fake_image = create_fake_image(100, 100)
                fake_image.save(save_path)
                return Path(save_path)
            else:
                save_path = os.path.join(tempfile.mkdtemp(), "output.png")
                output.save(save_path)
                return Path(save_path)
        if job_id:
            if not isinstance(output, Image.Image):
                if output.dtype != np.uint8:
                    output = output.astype(np.uint8)
                # Assuming the image was in BGR format, convert it to RGB.
                if output.shape[-1] == 3:  # Check if the image has three channels
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                output = Image.fromarray(output)
            print("Upload image")
            uploaded_image_url = upload_pil_image(output, "image/png")
            print("Send url to ai service for saving")
            post_to_ai_service(job_id, uploaded_image_url, callback_url)
            print("Define the output path")
            save_path = os.path.join(tempfile.mkdtemp(), "output.png")
            print("Create fake image, since we did the upload")
            fake_image = create_fake_image(100, 100)
            fake_image.save(save_path)
            return Path(save_path)
        else:
            save_path = os.path.join(tempfile.mkdtemp(), "output.png")
            cv2.imwrite(save_path, output)
            return Path(save_path)
