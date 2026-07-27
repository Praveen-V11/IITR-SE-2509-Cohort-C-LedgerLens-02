import io

from PIL import Image

from app.services import watermark


def test_watermark_output_is_valid_png_same_dimensions(tiny_png_bytes):
    out = watermark.apply_watermark(tiny_png_bytes, document_id="abc123-def456")
    img = Image.open(io.BytesIO(out))
    original = Image.open(io.BytesIO(tiny_png_bytes))
    assert img.size == original.size
    assert img.format == "PNG"


def test_watermark_changes_pixel_data(tiny_png_bytes):
    """A flat gray square should not come back byte-identical once a stamp
    has been drawn in the corner."""
    out = watermark.apply_watermark(tiny_png_bytes, document_id="abc123-def456")
    assert out != tiny_png_bytes
