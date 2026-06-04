"""
SAM2 Auto-Mask Demo

Compare SAM2.1 vs OpenCV auto-masking side-by-side.
Usage: python sam2_demo.py   → http://127.0.0.1:7861
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

from pipeline.auto_mask import auto_mask
import pipeline.auto_mask as _am  # access _am._am._SAM2_LOADED dynamically


def process(image: np.ndarray | None) -> tuple:
    """Run both auto-mask methods and return comparison."""
    if image is None:
        return None, None, None, 'Upload an image first.'

    h, w = image.shape[:2]
    if image.shape[-1] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
    else:
        rgb = image
        alpha = None

    # ---- SAM2 ----
    if _am._SAM2_LOADED:
        sam2_mask = auto_mask(rgb, prefer_sam=True)
    else:
        sam2_mask = None

    # ---- OpenCV ----
    cv_mask = auto_mask(rgb, prefer_sam=False)

    # ---- Build overlays ----
    def overlay(rgb, mask, color):
        """Draw mask contour in given color over the RGB image."""
        if mask is None:
            return rgb
        out = Image.fromarray(rgb).convert('RGBA')
        contour = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(contour)

        # Find mask boundary
        import cv2
        bin_mask = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            if len(pts) > 2:
                draw.polygon(pts, fill=(*color, 50), outline=(*color, 220), width=2)

        return Image.alpha_composite(out, contour).convert('RGB')

    sam2_overlay = overlay(rgb, sam2_mask, (0, 200, 80))  # green
    cv_overlay = overlay(rgb, cv_mask, (30, 140, 255))    # blue

    # ---- Status ----
    fg_pct_sam2 = (sam2_mask > 127).sum() / sam2_mask.size * 100 if sam2_mask is not None else 0
    fg_pct_cv = (cv_mask > 127).sum() / cv_mask.size * 100

    status = (
        f'**SAM2**: {"loaded" if _am._SAM2_LOADED else "not available (falling back to OpenCV)"}  |  '
        f'fg={fg_pct_sam2:.1f}%  |  '
        f'**OpenCV**: fg={fg_pct_cv:.1f}%'
    )

    return sam2_overlay, cv_overlay, Image.fromarray(sam2_mask) if sam2_mask is not None else None, status


# ---------------------------------------------------------------------------
#  UI
# ---------------------------------------------------------------------------
def build_ui():
    # Pre-load SAM2
    if _am._load_sam2():
        print('SAM2 ready for demo.')

    with gr.Blocks(title='SAM2 Auto-Mask Demo') as demo:
        gr.Markdown("""
        # SAM2.1 vs OpenCV — Auto-Mask Comparison

        SAM2.1 (Meta, Hiera-Small, 46M params) uses a centre-point prompt to segment the
        main object.  OpenCV uses border-colour estimation + Otsu thresholding.

        **Green contour** = SAM2.1   |   **Blue contour** = OpenCV
        """)

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label='Upload foreground image', type='numpy', height=320)

                with gr.Row():
                    btn = gr.Button('Run Auto-Mask', variant='primary', size='lg')

                # Also allow click-to-prompt (manual point for SAM2)
                gr.Markdown('*Click on the uploaded image* to set a manual prompt point for SAM2.')
                click_x = gr.State(0)
                click_y = gr.State(0)

                status = gr.Markdown('SAM2 loaded — upload and click Run.')

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab('Comparison'):
                        with gr.Row():
                            sam2_out = gr.Image(label='SAM2.1 (green contour)', type='pil', height=350)
                            cv_out = gr.Image(label='OpenCV (blue contour)', type='pil', height=350)
                    with gr.Tab('SAM2 Raw Mask'):
                        mask_out = gr.Image(label='SAM2 Binary Mask', type='pil', height=400)

        btn.click(fn=process, inputs=[inp],
                  outputs=[sam2_out, cv_out, mask_out, status])

        # Track click position on the input image
        def on_click(evt: gr.SelectData):
            return evt.index[0], evt.index[1]

        inp.select(on_click, outputs=[click_x, click_y])

    return demo


if __name__ == '__main__':
    demo = build_ui()
    demo.launch(server_name='0.0.0.0', server_port=7861, share=False)
