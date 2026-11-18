import requests
import base64
import os
from typing import List, Union
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, ViTModel, ResNetModel, ResNetConfig, AutoFeatureExtractor

AVAILABLE_ENCODERS = ["vit", "clip", "resnet"]
# DINOv2 retrieval encoders (selectable; not in the default list above).
# dinov2_last = CLS token after the final transformer block (12 of 12)
# dinov2_mid  = CLS token after block 6 of 12
DINOV2_ENCODERS = {"dinov2_last": 12, "dinov2_mid": 6}
DINOV2_ENCODERS.update({f"dinov2_b{k}": k for k in range(1, 13)})  # every block, for layer sweeps

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
RESNET_MODEL_NAME = "microsoft/resnet-50"
DINOV2_MODEL_NAME = "facebook/dinov2-base"  # ViT-B/14, 12 blocks, 768-d

# Models load on first use, so a run only downloads the encoders it selects.
_loaded = {}
def _load(kind):
    if kind not in _loaded:
        if kind == "clip":
            _loaded[kind] = (CLIPProcessor.from_pretrained(CLIP_MODEL_NAME), CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval())
        elif kind == "vit":
            _loaded[kind] = (AutoImageProcessor.from_pretrained(VIT_MODEL_NAME), ViTModel.from_pretrained(VIT_MODEL_NAME).eval())
        elif kind == "resnet":
            _loaded[kind] = (AutoFeatureExtractor.from_pretrained(RESNET_MODEL_NAME), ResNetModel.from_pretrained(RESNET_MODEL_NAME).eval())
        elif kind == "dinov2":
            _loaded[kind] = (AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME), AutoModel.from_pretrained(DINOV2_MODEL_NAME).eval())
    return _loaded[kind]

_dinov2_cache = {}  # image_path -> {encoder_name: L2-normalized CLS vector}
def _dinov2_embeddings(image_path, img):
    """One frozen forward pass; returns the CLS token at every layer in DINOV2_ENCODERS.

    hidden_states has 13 entries for the 12-block model: index 0 is the patch
    embedding output (before block 1) and index k is the output of block k.
    The model's final LayerNorm is applied to the CLS token at both depths
    (for block 12 this equals last_hidden_state[:, 0]), then each vector is
    L2-normalized.
    """
    if image_path not in _dinov2_cache:
        processor, model = _load("dinov2")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hs = outputs.hidden_states
            assert len(hs) == model.config.num_hidden_layers + 1 == 13
            out = {}
            for name, block in DINOV2_ENCODERS.items():
                cls = model.layernorm(hs[block])[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
                out[name] = cls.squeeze(0).tolist()
        _dinov2_cache[image_path] = out
    return _dinov2_cache[image_path]

def get_image_embedding(image_path: str, model_type: str = "clip") -> Union[List[float], dict]:
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB mode if it's not
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            if model_type.lower() in DINOV2_ENCODERS:
                return _dinov2_embeddings(image_path, img)[model_type.lower()]

            if model_type.lower() == "clip":
                clip_processor, clip_model = _load("clip")
                # Process the image using CLIP processor
                inputs = clip_processor(images=img, return_tensors="pt")
                
                # Get image features
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                
                # Convert to list and return
                return image_features.squeeze().tolist()
            
            elif model_type.lower() == "vit":
                vit_processor, vit_model = _load("vit")
                # Process the image using ViT processor
                inputs = vit_processor(images=img, return_tensors="pt")
                
                # Get image features
                with torch.no_grad():
                    outputs = vit_model(**inputs)
                
                # Use pooler_output as the image embedding
                image_embedding = outputs.pooler_output
                
                # Convert to list and return
                return image_embedding.squeeze().tolist()
            
            elif model_type.lower() == "resnet":
                resnet_feature_extractor, resnet_model = _load("resnet")
                # Process the image using ResNet feature extractor
                inputs = resnet_feature_extractor(images=img, return_tensors="pt")
                
                # Get image features
                with torch.no_grad():
                    outputs = resnet_model(**inputs)
                
                # Use pooler_output as the image embedding
                image_embedding = outputs.pooler_output
                
                # Convert to list and return
                return image_embedding.squeeze().tolist()
            
            else:
                return {"error": f"Unsupported model type: {model_type}"}
    
    except IOError:
        return {"error": f"Unable to open or process the image at {image_path}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    # Usage
    image_path = "/Users/muhammadarbabarshad/Documents/Personal Data/GPT4o-with-sakib/Overview.png"

    # Get embedding using CLIP
    clip_embedding = get_image_embedding(image_path, model_type="clip")

    if isinstance(clip_embedding, dict) and "error" in clip_embedding:
        print(f"CLIP Error: {clip_embedding['error']}")
    else:
        print(f"CLIP Embedding shape: {len(clip_embedding)}")
        print(f"CLIP First few values: {clip_embedding[:5]}")

    # Get embedding using ViT
    vit_embedding = get_image_embedding(image_path, model_type="vit")

    if isinstance(vit_embedding, dict) and "error" in vit_embedding:
        print(f"ViT Error: {vit_embedding['error']}")
    else:
        print(f"ViT Embedding shape: {len(vit_embedding)}")
        print(f"ViT First few values: {vit_embedding[:5]}")

    # Get embedding using ResNet
    resnet_embedding = get_image_embedding(image_path, model_type="resnet")

    if isinstance(resnet_embedding, dict) and "error" in resnet_embedding:
        print(f"ResNet Error: {resnet_embedding['error']}")
    else:
        print(f"ResNet Embedding shape: {len(resnet_embedding)}")
        print(f"ResNet First few values: {resnet_embedding[:5]}")

    # Note: This implementation uses the CLIP and ViT models directly from Hugging Face Transformers,
    # which is more efficient and doesn't require making API calls.
    # Make sure to install the required packages:
    # pip install transformers torch Pillow