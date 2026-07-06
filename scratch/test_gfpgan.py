import sys
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

try:
    from gfpgan import GFPGANer
    print("GFPGAN imported successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()

