import os
import sys
import numpy as np
import cv2
import torch

# Add workspace to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ImageUpscaler

def test():
    # 1. Create a dummy image
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    os.makedirs('scratch', exist_ok=True)
    cv2.imwrite('scratch/dummy.png', img)
    
    # 2. Initialize upscaler
    upscaler = ImageUpscaler(models_dir='.models')
    
    # 3. Test realesrgan-general (4x)
    print("Testing realesrgan-general...")
    try:
        upscaler.upscale(
            input_image_path='scratch/dummy.png',
            output_image_path='scratch/dummy_upscaled.png',
            model_key='realesrgan-general',
            tile_size=64, # small tiles for testing tiling code path
            denoise_level='none',
            target_scale=4,
            progress_callback=lambda msg, prog: print(f"  [{prog}%] {msg}")
        )
        print("realesrgan-general success!")
    except Exception as e:
        print("realesrgan-general FAILED!")
        import traceback
        traceback.print_exc()

    # 4. Test detail-enhance (1x)
    print("\nTesting detail-enhance...")
    try:
        upscaler.upscale(
            input_image_path='scratch/dummy.png',
            output_image_path='scratch/dummy_enhanced.png',
            model_key='detail-enhance',
            tile_size=64,
            denoise_level='none',
            target_scale=1,
            progress_callback=lambda msg, prog: print(f"  [{prog}%] {msg}")
        )
        print("detail-enhance success!")
    except Exception as e:
        print("detail-enhance FAILED!")
        import traceback
        traceback.print_exc()

    # 5. Test gfpgan-1.4
    print("\nTesting gfpgan-1.4...")
    try:
        upscaler.upscale(
            input_image_path='scratch/dummy.png',
            output_image_path='scratch/dummy_restored.png',
            model_key='gfpgan-1.4',
            tile_size=0,
            denoise_level='none',
            target_scale=1,
            progress_callback=lambda msg, prog: print(f"  [{prog}%] {msg}")
        )
        print("gfpgan-1.4 success!")
    except Exception as e:
        print("gfpgan-1.4 FAILED!")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test()
