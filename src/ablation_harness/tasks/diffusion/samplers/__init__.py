from .ddim import DDIMSampler
from .ddpm import DDPMSampler

SAMPLERS = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
}
