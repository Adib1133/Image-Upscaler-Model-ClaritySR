import sys
import torchvision.transforms.functional as transform_F
sys.modules['torchvision.transforms.functional_tensor'] = transform_F

import os
import requests
import threading
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2

# ==========================================
# Standalone PyTorch RRDBNet Architecture
# (Supports Real-ESRGAN, BSRGAN, ESRGAN)
# ==========================================

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) with 5 conv layers."""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)."""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def make_layer(basic_block, num_basic_block, **kwarg):
    """Stack basic blocks sequentially."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle for spatial downsampling + channel expansion."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class RRDBNet(nn.Module):
    """RRDBNet architecture supporting x1, x2, x4 native scales.
    Matches Real-ESRGAN / ESRGAN / BSRGAN weight formats.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Double upsample (2x * 2x = 4x native)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)
        out = self.upsampler(out)
        return out


# ==========================================
# Image Upscaler Pipeline Wrapper
# ==========================================

class ImageUpscaler:
    """Manages model loading, weight downloading, preprocessing, tiling, and inference."""

    MODEL_CONFIGS = {
        # ---- Real-ESRGAN Family ----
        'realesrgan-general': {
            'name': 'RealESRGAN_x4plus.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'scale': 4,
            'num_block': 23,
            'arch': 'rrdbnet',
            'desc': 'Real-ESRGAN x4+ (General Photos)'
        },
        'realesrgan-anime': {
            'name': 'RealESRGAN_x4plus_anime_6B.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'scale': 4,
            'num_block': 6,
            'arch': 'rrdbnet',
            'desc': 'Real-ESRGAN x4+ Anime (Illustrations / Fast)'
        },
        'realesrgan-x2': {
            'name': 'RealESRGAN_x2plus.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            'scale': 2,
            'num_block': 23,
            'arch': 'rrdbnet',
            'desc': 'Real-ESRGAN x2+ (General Photos / 2x)'
        },
        # ---- BSRGAN ----
        'bsrgan-x4': {
            'name': 'BSRGAN.pth',
            'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth',
            'scale': 4,
            'num_block': 23,
            'arch': 'rrdbnet',
            'desc': 'BSRGAN x4 (De-blurring / De-noising)'
        },
        # ---- Classic ESRGAN (Realistic textures) ----
        'esrgan-classic': {
            'name': 'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
            'scale': 4,
            'num_block': 23,
            'arch': 'rrdbnet',
            'desc': 'ESRGAN Classic x4 (Sharp / Realistic Textures)'
        },
        # ---- Real-ESRGAN General v3 (sharper output) ----
        'realesrgan-v3': {
            'name': 'realesr-general-x4v3.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            'scale': 4,
            'num_block': 6,
            'arch': 'srvggnetcompact',
            'desc': 'Real-ESRGAN v3 (General / Balanced Quality)'
        },
        # ---- GFPGAN ----
        'gfpgan-1.4': {
            'name': 'GFPGANv1.4.pth',
            'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
            'scale': 1,
            'num_block': 0,
            'arch': 'gfpgan',
            'desc': 'GFPGAN v1.4 (Face & Artifact Restoration)'
        },
        # ---- Detail Enhancement Only (1x, no upscaling) ----
        'detail-enhance': {
            'name': 'realesr-general-x4v3.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            'scale': 4,
            'num_block': 6,
            'arch': 'srvggnetcompact',
            'desc': '1x Detail Enhance (Real-ESRGAN v3 Backbone)'
        },
    }

    def __init__(self, models_dir='.models'):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SYSTEM] Hardware environment initialization: using {self.device.type.upper()}")
        self.loaded_model_name = None
        self.model = None

    def _load_weights(self, model_path):
        try:
            return torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(model_path, map_location=self.device)

    def download_weights(self, model_key, progress_callback=None, stop_event=None):
        """Downloads model weights with progress tracking."""
        config = self.MODEL_CONFIGS.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")

        if config['name'] is None:
            return None  # No weights needed

        model_path = os.path.join(self.models_dir, config['name'])
        if os.path.exists(model_path):
            if progress_callback:
                progress_callback("Weights already cached.", 5)
            return model_path

        url = config['url']
        if progress_callback:
            progress_callback(f"Downloading {config['name']}...", 0)

        temp_path = f"{model_path}.part"
        response = requests.get(url, stream=True, timeout=(10, 120))
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if stop_event and stop_event.is_set():
                    f.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise InterruptedError("Download cancelled by user.")
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        pct = int((downloaded / total_size) * 15)  # Download takes first 15%
                        progress_callback(
                            f"Downloading {config['name']}: {downloaded // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB",
                            pct
                        )

        if progress_callback:
            progress_callback("Download complete.", 15)
        os.replace(temp_path, model_path)
        return model_path

    def load_model(self, model_key, progress_callback=None, stop_event=None):
        """Loads a model from the cached weight file into RAM/GPU memory."""
        if self.loaded_model_name == model_key and self.model is not None:
            if progress_callback:
                progress_callback("Model already loaded.", 20)
            return

        config = self.MODEL_CONFIGS.get(model_key)
        if not config:
            raise ValueError(f"Unknown model key: {model_key}")

        arch = config.get('arch', 'rrdbnet')

        model_path = self.download_weights(model_key, progress_callback, stop_event)

        if stop_event and stop_event.is_set():
            raise InterruptedError("Cancelled before model load.")

        if arch == 'gfpgan':
            if progress_callback:
                progress_callback("Initializing GFPGAN...", 16)
            from gfpgan import GFPGANer
            self.model = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            self.loaded_model_name = model_key
            if progress_callback:
                progress_callback(f"GFPGAN loaded.", 20)
            return

        if progress_callback:
            progress_callback("Loading model into memory...", 16)

        if arch == 'srvggnetcompact':
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=config['scale'],
                act_type='prelu'
            )
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                scale=config['scale'],
                num_feat=64,
                num_block=config['num_block'],
                num_grow_ch=32
            )

        if progress_callback:
            progress_callback("Parsing weights state dict...", 18)

        state_dict = self._load_weights(model_path)

        # Real-ESRGAN stores weights in 'params' or 'params_ema'
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']

        # Map legacy ESRGAN/BSRGAN key naming to BasicSR structure (only for RRDBNet models)
        if arch != 'srvggnetcompact':
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith('RRDB_trunk.'):
                    new_k = k.replace('RRDB_trunk.', 'body.')
                    new_k = new_k.replace('.RDB1.', '.rdb1.')
                    new_k = new_k.replace('.RDB2.', '.rdb2.')
                    new_k = new_k.replace('.RDB3.', '.rdb3.')
                elif k.startswith('trunk_conv.'):
                    new_k = k.replace('trunk_conv.', 'conv_body.')
                elif k.startswith('upconv1.'):
                    new_k = k.replace('upconv1.', 'conv_up1.')
                elif k.startswith('upconv2.'):
                    new_k = k.replace('upconv2.', 'conv_up2.')
                elif k.startswith('HRconv.'):
                    new_k = k.replace('HRconv.', 'conv_hr.')
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(self.device)

        self.model = model
        self.loaded_model_name = model_key

        if progress_callback:
            progress_callback(f"Model loaded on {self.device.type.upper()}.", 20)

    def denoise_image(self, img_np, level):
        """Applies pre-upscale denoising filters."""
        if level == 'mild':
            return cv2.bilateralFilter(img_np, d=5, sigmaColor=35, sigmaSpace=35)
        elif level == 'medium':
            return cv2.bilateralFilter(img_np, d=7, sigmaColor=50, sigmaSpace=50)
        elif level == 'strong':
            denoised = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
            return cv2.GaussianBlur(denoised, (3, 3), 0)
        return img_np

    def upscale(self, input_image_path, output_image_path,
                model_key='realesrgan-general', tile_size=512,
                denoise_level='none', target_scale=4,
                progress_callback=None, stop_event=None):
        """Runs upscaling on an image with tiling, denoising, and progress reporting."""

        def check_stop():
            if stop_event and stop_event.is_set():
                raise InterruptedError("Upscaling stopped by user.")

        # Step 1: Load / verify model
        self.load_model(model_key, progress_callback, stop_event)
        check_stop()

        config = self.MODEL_CONFIGS[model_key]
        scale = config['scale']
        arch = config.get('arch', 'rrdbnet')

        if progress_callback:
            progress_callback("Preprocessing input image...", 20)

        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image at {input_image_path}")

        orig_h, orig_w = img.shape[:2]

        # Dynamically adjust tile size based on available system memory / VRAM
        import psutil
        def get_avail_mem():
            if self.device.type == 'cuda':
                try:
                    free_m, _ = torch.cuda.mem_get_info(self.device)
                    return free_m
                except RuntimeError:
                    pass
            return psutil.virtual_memory().available

        avail_mem = get_avail_mem()
        pixel_multiplier = 12000
        max_allowed_mem = int(avail_mem * 0.8)
        
        effective_tile_size = tile_size
        if effective_tile_size == 0:
            estimated_mem = orig_h * orig_w * pixel_multiplier
            if estimated_mem > max_allowed_mem:
                for ts in [512, 256, 128]:
                    if ts * ts * pixel_multiplier < max_allowed_mem:
                        effective_tile_size = ts
                        break
                else:
                    effective_tile_size = 128
                if progress_callback:
                    progress_callback(f"Low memory alert ({avail_mem // (1024*1024)}MB free). Auto-enabling tiling ({effective_tile_size}px) to prevent crash.", 21)
        elif effective_tile_size > 0:
            estimated_mem = effective_tile_size * effective_tile_size * pixel_multiplier
            if estimated_mem > max_allowed_mem:
                for ts in [256, 128, 64]:
                    if ts * ts * pixel_multiplier < max_allowed_mem:
                        effective_tile_size = ts
                        break
                else:
                    effective_tile_size = 64
                if progress_callback:
                    progress_callback(f"Low memory alert. Reducing tile size to {effective_tile_size}px to prevent crash.", 21)

        tile_size = effective_tile_size

        # Pre-denoising
        if denoise_level != 'none':
            if progress_callback:
                progress_callback("Applying denoising filters...", 21)
            img = self.denoise_image(img, denoise_level)

        check_stop()

        if arch == 'gfpgan':
            if progress_callback:
                progress_callback("Running GFPGAN restoration pass...", 25)
            _, _, out_bgr = self.model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            if progress_callback:
                progress_callback("Postprocessing GFPGAN image...", 90)
        else:
            # Convert BGR to RGB and normalize to [0, 1].
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
    
            # Tensor: (1, 3, H, W)
            x = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
            B, C, H, W = x.shape
    
            # ---- 1x Detail Enhance Mode ----
            if target_scale == 1 and arch == 'rrdbnet':
                if progress_callback:
                    progress_callback("Running detail enhancement pass...", 25)
                check_stop()
                with torch.no_grad():
                    pad_h = (8 - H % 8) % 8
                    pad_w = (8 - W % 8) % 8
                    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

                    # Use the RRDBNet at 4x then downsample back to 1x
                    out_4x = self.model(x_pad)
                    out = F.interpolate(out_4x[:, :, :H * 4, :W * 4],
                                        size=(H, W), mode='bicubic', align_corners=False)

                out = out[:, :, :H, :W]
                if progress_callback:
                    progress_callback("Postprocessing detail-enhanced image...", 90)
            else:
                # ---- Normal Tiled / Full Upscaling Mode ----
                tile_pad = 32
                use_tiles = tile_size > 0 and (H > tile_size or W > tile_size)

                if not use_tiles:
                    if progress_callback:
                        progress_callback("Upscaling full image (no tiling)...", 22)
                    check_stop()
                    with torch.no_grad():
                        pad_h = (4 - H % 4) % 4
                        pad_w = (4 - W % 4) % 4
                        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
                        output_tensor = self.model(x_padded)
                        output_tensor = output_tensor[:, :, :H * scale, :W * scale]
                    if progress_callback:
                        progress_callback("Postprocessing upscaled tensor...", 88)
                    out = output_tensor
                else:
                    if progress_callback:
                        progress_callback("Initializing tiled inference...", 22)

                    x_padded = F.pad(x, (tile_pad, tile_pad, tile_pad, tile_pad), mode='replicate')
                    output_tensor = torch.zeros((1, C, H * scale, W * scale), device=self.device)

                    num_h_tiles = (H + tile_size - 1) // tile_size
                    num_w_tiles = (W + tile_size - 1) // tile_size
                    total_tiles = num_h_tiles * num_w_tiles
                    processed_tiles = 0

                    for h_start in range(0, H, tile_size):
                        h_end = min(h_start + tile_size, H)
                        for w_start in range(0, W, tile_size):
                            check_stop()
                            w_end = min(w_start + tile_size, W)

                            if progress_callback:
                                pct = int(22 + (processed_tiles / total_tiles) * 65)
                                progress_callback(
                                    f"Processing tile {processed_tiles + 1}/{total_tiles}...", pct
                                )

                            patch_h_start = h_start
                            patch_h_end = h_end + 2 * tile_pad
                            patch_w_start = w_start
                            patch_w_end = w_end + 2 * tile_pad

                            tile_patch = x_padded[:, :, patch_h_start:patch_h_end, patch_w_start:patch_w_end]

                            _, _, patch_h, patch_w = tile_patch.shape
                            pad_h_t = (4 - patch_h % 4) % 4
                            pad_w_t = (4 - patch_w % 4) % 4
                            if pad_h_t > 0 or pad_w_t > 0:
                                tile_patch = F.pad(tile_patch, (0, pad_w_t, 0, pad_h_t), mode='replicate')

                            with torch.no_grad():
                                tile_out = self.model(tile_patch)

                            tile_out = tile_out[:, :, :patch_h * scale, :patch_w * scale]

                            crop_h_start = tile_pad * scale
                            crop_h_end = crop_h_start + (h_end - h_start) * scale
                            crop_w_start = tile_pad * scale
                            crop_w_end = crop_w_start + (w_end - w_start) * scale

                            tile_out_cropped = tile_out[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
                            output_tensor[:, :, h_start * scale:h_end * scale, w_start * scale:w_end * scale] = tile_out_cropped

                            processed_tiles += 1

                    if progress_callback:
                        progress_callback("Stitching tiles complete.", 90)
                    out = output_tensor

            check_stop()

            # Convert tensor to uint8 numpy BGR.
            out_np = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_np = (out_np * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

        # Resize to user-requested target scale (if it differs from model's native output)
        target_h = int(orig_h * target_scale)
        target_w = int(orig_w * target_scale)
        if out_bgr.shape[0] != target_h or out_bgr.shape[1] != target_w:
            if progress_callback:
                progress_callback(f"Resizing to target scale ({target_scale}x) via Lanczos...", 93)
            out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if progress_callback:
            progress_callback("Saving output image to disk...", 97)

        if not cv2.imwrite(output_image_path, out_bgr):
            raise OSError(f"Could not save output image to {output_image_path}")

        if progress_callback:
            progress_callback("Upscaling finished successfully!", 100)

        return True
