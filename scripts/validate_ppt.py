"""校验 PPTX 的幻灯片数量、标题、备注与图片数量。"""
from pathlib import Path

from pptx import Presentation

path = Path(r"C:\Users\Nakan\Desktop\航拍建筑物提取_论文汇报.pptx")
prs = Presentation(path)
print("slides:", len(prs.slides))
print("size:", prs.slide_width.inches, "x", prs.slide_height.inches)
for i, slide in enumerate(prs.slides, 1):
    texts = [sh.text.strip().replace("\n", " / ") for sh in slide.shapes if sh.has_text_frame and sh.text.strip()]
    title = texts[0] if texts else "(no text)"
    pics = sum(1 for sh in slide.shapes if sh.shape_type == 13)
    notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
    print(f"{i:02d} | pics={pics} | notes={len(notes)} | {title[:60]}")
