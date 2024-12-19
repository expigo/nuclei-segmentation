from .unet.base import UNet
from .unet.vanilla_unet import VanillaUNet
from .unet.smp_unet import SMPUNet

MODEL_REGISTRY = {
    'vanilla_unet': VanillaUNet,
    'smp_unet': SMPUNet,
}