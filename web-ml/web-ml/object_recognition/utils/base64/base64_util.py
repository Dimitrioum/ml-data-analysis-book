import io
import base64 as b64
import PIL as pil
from numpy import asarray


def base64_encode(img_numpy):
    if img_numpy is None:
        return ''
    img = pil.Image.fromarray(img_numpy)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_byte = buf.getvalue()
    return b64.b64encode(img_byte)


def base64_decode(img_base64):
    img_pil = pil.Image.open(
        io.BytesIO(
            b64.b64decode(img_base64)
        )
    )
    return asarray(img_pil)
