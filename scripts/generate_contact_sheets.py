"""Generate concise visual contact sheets for GWLab Phase 0 concepts."""

import os
from PIL import Image, ImageDraw, ImageFont

SCREENSHOTS_DIR = "reports/screenshots/concepts"

def create_contact_sheet(images_info, output_path, title):
    # images_info: list of tuples (img_path, label_text)
    # Thumbnail target size: 720 x 450 (preserving 16:10 aspect ratio)
    thumb_w, thumb_h = 720, 450
    header_h = 60
    card_header_h = 36
    margin = 20
    
    n = len(images_info)
    total_w = n * thumb_w + (n + 1) * margin
    total_h = header_h + card_header_h + thumb_h + 2 * margin
    
    sheet = Image.new("RGB", (total_w, total_h), color=(18, 21, 26))
    draw = ImageDraw.Draw(sheet)
    
    # Try default fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_label = ImageFont.truetype("arial.ttf", 16)
        font_sub = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    # Draw title
    draw.text((margin, 16), title, fill=(240, 242, 245), font=font_title)
    draw.text((total_w - 360, 22), "GWLab Phase 0 · 1440x900 Desktop", fill=(140, 150, 165), font=font_sub)
    draw.line([(margin, header_h - 10), (total_w - margin, header_h - 10)], fill=(45, 52, 65), width=1)
    
    for i, (rel_path, label) in enumerate(images_info):
        full_path = os.path.join(SCREENSHOTS_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found!")
            continue
            
        im = Image.open(full_path)
        im_thumb = im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        
        x = margin + i * (thumb_w + margin)
        y = header_h + card_header_h
        
        # Draw card header
        draw.text((x, header_h + 8), label, fill=(230, 94, 46), font=font_label)
        
        # Draw border and paste image
        draw.rectangle([x - 1, y - 1, x + thumb_w, y + thumb_h], outline=(60, 70, 85), width=1)
        sheet.paste(im_thumb, (x, y))
        
    sheet.save(output_path, "PNG", optimize=True)
    print(f"Saved contact sheet to {output_path} ({total_w}x{total_h}px)")

def main():
    home_images = [
        ("concept_a_hero_desktop_1440x900_dark.png", "Concept A: Scientific Journal (Hero)"),
        ("concept_b_hero_desktop_1440x900_dark.png", "Concept B: Instrument Laboratory (Hero)"),
        ("concept_c_hero_desktop_1440x900_dark.png", "Concept C: Contemporary Physics Atlas (Hero)"),
    ]
    create_contact_sheet(
        home_images,
        os.path.join(SCREENSHOTS_DIR, "contact_sheet_home.png"),
        "GWLab Phase 0 · Home Concepts Contact Sheet (Dark Theme)"
    )
    
    waveforms_images = [
        ("concept_a_waveforms_desktop_1440x900_dark.png", "Concept A: Scientific Journal (Waveforms)"),
        ("concept_b_waveforms_desktop_1440x900_dark.png", "Concept B: Instrument Laboratory (Waveforms)"),
        ("concept_c_waveforms_desktop_1440x900_dark.png", "Concept C: Contemporary Physics Atlas (Waveforms)"),
    ]
    create_contact_sheet(
        waveforms_images,
        os.path.join(SCREENSHOTS_DIR, "contact_sheet_waveforms.png"),
        "GWLab Phase 0 · Waveforms Concepts Contact Sheet (Dark Theme)"
    )

if __name__ == "__main__":
    main()
