IS_INSIGHTFACE_INSTALLED = False
try:
    from insightface.app import FaceAnalysis
    IS_INSIGHTFACE_INSTALLED = True
except ImportError:
    pass

if not IS_INSIGHTFACE_INSTALLED:
    raise Exception("Please install insightface to use this node.")

INSTALLED_LIBRARIES = []
if IS_INSIGHTFACE_INSTALLED:
    INSTALLED_LIBRARIES.append("insightface")

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import comfy.utils
import os
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

DLIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlib")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "library": (INSTALLED_LIBRARIES, ),
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], ),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceAnalysis"

    def load_models(self, library, provider):
        out = {}

        if library == "insightface":
            out = {
                "library": library,
                "detector": FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
            }
            out["detector"].prepare(ctx_id=0, det_size=(640, 640))
        else:
            out = {
                "library": library,
                "detector": dlib.get_frontal_face_detector(),
                "shape_predict": dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat")),
                "face_recog": dlib.face_recognition_model_v1(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat")),
            }
        

        return (out, )

def crop_face(image, x, y, w, h, padding=0):
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.width, w + 2 * padding)
    h = min(image.height, h + 2 * padding)

    return image.crop((x, y, x + w, y + h))

class FaceBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "index": ("INT", { "default": -1, "min": -1, "max": 4096, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "x", "y", "width", "height")
    FUNCTION = "bbox"
    CATEGORY = "FaceAnalysis"
    OUTPUT_IS_LIST = (True, True, True, True, True,)

    def bbox(self, analysis_models, image, padding, index=-1):
        out_img = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for i in image:
            img = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')

            if analysis_models["library"] == "insightface":
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

            else:
                faces = analysis_models["detector"](np.array(img), 1)
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
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

        if len(out_img) == 1:
            index = 0

        if index > len(out_img) - 1:
            index = len(out_img) - 1

        if index != -1:
            out_img = [out_img[index]]
            out_x = [out_x[index]]
            out_y = [out_y[index]]
            out_w = [out_w[index]]
            out_h = [out_h[index]]
        #else:
        #    w = out_img[0].shape[1]
        #    h = out_img[0].shape[0]

            #out_img = [comfy.utils.common_upscale(img.unsqueeze(0).movedim(-1,1), w, h, "bilinear", "center").movedim(1,-1).squeeze(0) for img in out_img]
            #out_img = torch.stack(out_img)
        
        return (out_img, out_x, out_y, out_w, out_h,)
