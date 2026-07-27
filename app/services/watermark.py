"""
Provenance watermarking. Every archived source image gets a visible stamp
(document id + UTC timestamp) burned into the corner before it's written
to disk, so an archived image can never be mistaken for an unmodified
original later on.
"""

import datetime as dt
import io

from PIL import Image, ImageDraw, ImageFont

WATERMARK_MARGIN = 12
WATERMARK_OPACITY = 160  # out of 255


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        # DejaVuSans isn't guaranteed to be on every base image - fall back
        # to PIL's built-in bitmap font rather than crashing the pipeline
        # over a cosmetic detail.
        return ImageFont.load_default()


def apply_watermark(image_bytes: bytes, document_id: str) -> bytes:
    stamp = f"LedgerLens \u00b7 {document_id[:8]} \u00b7 {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}Z"

    base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _load_font()
    text_bbox = draw.textbbox((0, 0), stamp, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    x = base.width - text_w - WATERMARK_MARGIN
    y = base.height - text_h - WATERMARK_MARGIN

    # Faint backing rectangle so the stamp stays legible over busy receipt
    # backgrounds (a plain text overlay disappears against printed text).
    draw.rectangle(
        [x - 6, y - 4, x + text_w + 6, y + text_h + 8],
        fill=(0, 0, 0, WATERMARK_OPACITY // 2),
    )
    draw.text((x, y), stamp, font=font, fill=(255, 255, 255, WATERMARK_OPACITY))

    watermarked = Image.alpha_composite(base, overlay).convert("RGB")

    out = io.BytesIO()
    watermarked.save(out, format="PNG")
    return out.getvalue()
