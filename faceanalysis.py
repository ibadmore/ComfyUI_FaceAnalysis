IS_INSIGHTFACE_INSTALLED = False
try:
    from insightface.app import FaceAnalysis
    IS_INSIGHTFACE_INSTALLED = True
except ImportError:
    pass

if not IS_INSIGHTFACE_INSTALLED:
    raise Exception("Please install insightface to use this node.")

INSTALLED_LIBRARIES = ["insightface"] if IS_INSIGHTFACE_INSTALLED else []

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import comfy.utils
import os
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "library": (INSTALLED_LIBRARIES, ),
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], ),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceAnalysis"

    def load_models(self, library, provider):
        out = {
            "library": library,
            "detector": FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
        }
        out["detector"].prepare(ctx_id=0, det_size=(640, 640))
        return (out, )

class FaceBoundingBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "index": ("INT", { "default": -1, "min": -1, "max": 4096, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    FUNCTION = "bbox"
    CATEGORY = "FaceAnalysis"

    def bbox(self, analysis_models, image, padding, index=-1):
        out_img = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for i in image:
            img = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
            faces = analysis_models["detector"].get(np.array(img))
            for face in faces:
                x, y, w, h = face.bbox.astype(int)
                w = w - x
                h = h - y
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.width, w + 2 * padding)
                h = min(img.height, h + 2 * padding)
                crop = img.crop((x, y, x + w, y + h))
                out_img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
                out_x.append(x)
                out_y.append(y)
                out_w.append(w)
                out_h.append(h)

        if not out_img:
            raise Exception('No face detected in image.')

        if index >= 0 and index < len(out_img):
            out_img = [out_img[index]]
            out_x = [out_x[index]]
            out_y = [out_y[index]]
            out_w = [out_w[index]]
            out_h = [out_h[index]]
        
        return (out_img, out_x, out_y, out_w, out_h,)

NODE_CLASS_MAPPINGS = {
    "FaceAnalysisModels": FaceAnalysisModels,
    "FaceBoundingBox": FaceBoundingBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAnalysisModels": "Face Analysis Models",
    "FaceBoundingBox": "Face Bounding Box",
}
