import io
import os

import torch
import numpy as np
import boto3
from PIL import Image, ImageSequence, ImageOps

from nodes.logger import logger

from dotenv import load_dotenv

load_dotenv()


def get_content_type(filename):
    content_type_mapping = {
        ".png": "image/png",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
    }
    _, file_extension = os.path.splitext(filename)
    return content_type_mapping.get(file_extension.lower(), "binary/octet-stream")


def awss3_save_file(client, bucket, key, buff, acl="public-read"):
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buff,
        ACL=acl,
        ContentType=get_content_type(key),
    )


def awss3_load_file(client, bucket, key):
    outfile = io.BytesIO()
    client.download_fileobj(bucket, key, outfile)
    outfile.seek(0)
    return outfile


def awss3_init_client():
    ak = os.getenv("AWS_ACCESS_KEY_ID", None)
    sk = os.getenv("AWS_SECRET_ACCESS_KEY", None)
    region_name = os.getenv("AWS_REGION_NAME", None)

    if not all([ak, sk, region_name]):
        err = "Missing required S3 environment variables."
        logger.error(err)
    return boto3.client('s3', region_name=region_name, aws_access_key_id=ak, aws_secret_access_key=sk)


class ConvertPngToWebp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "resizing_width": ("INT", {"default": 200}),
                             "resizing_height": ("INT", {"default": 200})
                             }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_png_to_webp"
    CATEGORY = "iw-image-s3"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)

    def convert_to_thumbnail_webp(self, image, resizing_width, resizing_height):
        if isinstance(image, io.BytesIO):
            img_byte_arr = image
        else:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')

        image_loaded = Image.open(img_byte_arr)

        # 원본 이미지의 크기
        width, height = image_loaded.size
        # 중앙 정사각형 크기: 가로, 세로 중 작은 값
        square_side = min(width, height)

        # 중앙 정사각형 영역 계산
        left = (width - square_side) // 2
        top = (height - square_side) // 2
        right = left + square_side
        bottom = top + square_side

        # 중앙 정사각형으로 크롭
        cropped_image = image_loaded.crop((left, top, right, bottom))
        # 200x200 크기로 리사이즈
        resized_image = cropped_image.resize((resizing_width, resizing_height), Image.Resampling.LANCZOS)
        # WebP로 변환된 이미지를 담을 BytesIO 객체 생성
        webp_image = io.BytesIO()
        # WebP로 저장 (품질을 80으로 설정)
        resized_image.save(webp_image, format="WEBP", quality=80)

        # 파일 포인터를 처음으로 되돌림
        webp_image.seek(0)

        return (webp_image,)


# SaveImageToS3
class SaveImageToS3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                             "object_key": ("STRING", {"multiline": False, "default": "object_key"})
                             }}

    FUNCTION = "save_image_to_s3"
    CATEGORY = "iw-image-s3"
    OUTPUT_NODE = True

    def save_image_to_s3(self, image, s3_bucket, object_key):
        client = awss3_init_client()
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        awss3_save_file(client, s3_bucket, object_key, img_byte_arr)
        return {"ui": {"image": (image,)}}


# LoadImageFromS3
class LoadImageFromS3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
            "pathname": ("STRING", {"multiline": False, "default": "pathname for file"})
        }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "load_image_from_s3"
    CATEGORY = "iw-image-s3"

    def load_image_from_s3(self, s3_bucket, object_key):
        client = awss3_init_client()
        img = Image.open(awss3_load_file(client, s3_bucket, object_key))
        output_images = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

        return (output_image,)


if __name__ == '__main__':
    # 250405 aws 관련 테스트 성공
    # client = awss3_init_client()
    # awss3_save_file(client, "image-wizard-dev", "generation/zlVGHyAYREhrkUoIJFYtnj2TeTM2/508/output/HAIR_CHANGE_output_1_20250319154227_436091d5-560b-496e-9c22-ded81c160b60.png",
    #                 awss3_load_file(client, "image-wizard-dev", "generation/zlVGHyAYREhrkUoIJFYtnj2TeTM2/508/input/ORIGINAL_20250304132406_ec1ba610-de42-4ed4-af7f-f39713ef55d3.png"))

    # 250405 webp 관련 테스트 성공
    client = awss3_init_client()
    webp_converter = ConvertPngToWebp()
    webp_result = webp_converter.convert_to_thumbnail_webp(image=awss3_load_file(client, "image-wizard-dev",
                                                                                 "generation/zlVGHyAYREhrkUoIJFYtnj2TeTM2/508/input/ORIGINAL_20250304132406_ec1ba610-de42-4ed4-af7f-f39713ef55d3.png"),
                                                           resizing_width=200, resizing_height=200)[0]

    awss3_save_file(client, "image-wizard-dev", "generation/zlVGHyAYREhrkUoIJFYtnj2TeTM2/508/output/HAIR_CHANGE_output_1_20250319154227_436091d5-560b-496e-9c22-ded81c160b60.webp", webp_result)
