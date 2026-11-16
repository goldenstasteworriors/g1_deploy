"""Training-only lighting changes, shared across a sample's entire timeline.

No geometric transforms: original target coordinates, state and action stay valid.
Future video labels receive the same camera transform as observation/reference RGB.
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from lab_low.augmentations import sample_local_light, local_light_field


def sample_lighting(config, rng):
    if not config or not config.get('enabled', False) or rng.random() >= float(config['probability']):
        return None
    def draw(name):
        return float(rng.uniform(*config[name]))
    parameters = dict(brightness=draw('brightness'), contrast=draw('contrast'),
                saturation=draw('saturation'), gamma=draw('gamma'),
                white_balance=rng.uniform(*config['white_balance'], size=3),
                gradients=rng.uniform(-float(config['illumination_gradient']),
                                      float(config['illumination_gradient']), size=(3, 2)),
                blur=draw('blur_radius') if rng.random() < float(config['blur_probability']) else 0.)
    parameters['local_fields'] = sample_local_light(config.get('local'), rng)
    return parameters


def apply_lighting(image, params, camera):
    if params is None:
        return image
    image = ImageEnhance.Contrast(image).enhance(params['contrast'])
    image = ImageEnhance.Color(image).enhance(params['saturation'])
    a = np.asarray(image, dtype=np.float32) / 255.
    a = np.power(a, params['gamma'])
    h, w = a.shape[:2]
    gx, gy = params['gradients'][camera]
    field = 1 + gx*np.linspace(-1, 1, w, dtype=np.float32)[None, :] + gy*np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    a *= (params['brightness'] * params['white_balance']).astype(np.float32)[None, None, :]
    a *= field[:, :, None]
    if params.get('local_fields') is not None:
        a *= local_light_field(image.size, params['local_fields'][camera])[:, :, None]
    image = Image.fromarray(np.uint8(np.clip(a, 0, 1)*255))
    if params['blur']:
        image = image.filter(ImageFilter.GaussianBlur(params['blur']))
    return image
