"""生成桌面上的论文汇报 PPT。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(r"D:\All4DS\BISHE")
OUT = Path(r"C:\Users\Nakan\Desktop\航拍建筑物提取_论文汇报.pptx")

NAVY = RGBColor(0x10, 0x27, 0x40)
TEAL = RGBColor(0x1F, 0x6F, 0x8B)
TEAL_L = RGBColor(0x8F, 0xBF, 0xCF)
ORANGE = RGBColor(0xE0, 0x7A, 0x5F)
GREEN = RGBColor(0x3A, 0x7D, 0x44)
GRAY = RGBColor(0x5B, 0x65, 0x73)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
MID = RGBColor(0xD9, 0xE2, 0xEA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BG = RGBColor(0xFB, 0xFC, 0xFE)

FONT = "Microsoft YaHei"
FONT_EN = "Arial"


def set_fill(shape, color, alpha=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if alpha is not None:
        shape.fill.transparency = alpha


def no_line(shape):
    shape.line.fill.background()


def add_text(slide, text, x, y, w, h, size=14, bold=False, color=NAVY,
             align=PP_ALIGN.LEFT, font=FONT, spacing=1.08, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box


def add_bullets(slide, items, x, y, w, h, size=12, color=GRAY, spacing=1.12, bullet_color=TEAL):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = spacing
        p.space_after = Pt(6)
        r1 = p.add_run()
        r1.text = "•  "
        r1.font.size = Pt(size)
        r1.font.color.rgb = bullet_color
        r1.font.bold = True
        r1.font.name = FONT
        r2 = p.add_run()
        r2.text = item
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
        r2.font.name = FONT
    return box


def add_rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    set_fill(shp, fill)
    if line is None:
        no_line(shp)
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    return shp


def add_card(slide, x, y, w, h, title, body, accent=TEAL, fill=LIGHT, title_size=13, body_size=10.5):
    add_rect(slide, x, y, w, h, fill, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_rect(slide, x, y + 0.08, 0.06, h - 0.16, accent)
    add_text(slide, title, x + 0.22, y + 0.14, w - 0.35, 0.32, title_size, True, NAVY)
    add_text(slide, body, x + 0.22, y + 0.50, w - 0.35, h - 0.58, body_size, False, GRAY, spacing=1.12)


def add_picture_fit(slide, path, x, y, max_w, max_h, border=False):
    with Image.open(path) as im:
        ar = im.width / im.height
    if ar >= max_w / max_h:
        w = max_w
        h = w / ar
    else:
        h = max_h
        w = h * ar
    left = x + (max_w - w) / 2
    top = y + (max_h - h) / 2
    if border:
        add_rect(slide, left - 0.035, top - 0.035, w + 0.07, h + 0.07, WHITE, line=MID)
    pic = slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(w), Inches(h))
    return pic, left, top, w, h


def add_table(slide, data, x, y, w, col_widths, row_h=0.36, header=True, font_size=10.5):
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows))
    table = shape.table
    for i, cw in enumerate(col_widths):
        table.columns[i].width = Inches(cw)
    for r in range(rows):
        table.rows[r].height = Inches(row_h)
        for c in range(cols):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if header and r == 0:
                cell.fill.fore_color.rgb = NAVY
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 == 1 else LIGHT
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(data[r][c])
            run.font.size = Pt(font_size)
            run.font.name = FONT
            run.font.bold = header and r == 0
            run.font.color.rgb = WHITE if (header and r == 0) else NAVY
    return table


def add_header(slide, title, subtitle=None, page=None):
    add_rect(slide, 0, 0, 0.12, 7.5, TEAL)
    add_text(slide, title, 0.55, 0.30, 12.2, 0.55, 24, True, NAVY)
    if subtitle:
        add_text(slide, subtitle, 0.57, 0.90, 12.1, 0.34, 11.5, False, GRAY)
    add_rect(slide, 0.55, 1.28 if subtitle else 1.02, 12.2, 0.018, MID)
    add_text(slide, "基于深度学习的航拍图像建筑物自动提取", 0.55, 7.12, 7.5, 0.25, 8.5, False, GRAY)
    if page is not None:
        add_text(slide, f"{page:02d}", 12.25, 7.08, 0.55, 0.28, 9, True, TEAL, PP_ALIGN.RIGHT)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------------- 1. 封面 ----------------
    s = prs.slides.add_slide(blank)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY
    add_rect(s, 0, 0, 0.16, 7.5, TEAL)
    # 右侧航拍图
    add_rect(s, 8.38, 1.03, 4.36, 4.36, RGBColor(0x1B, 0x35, 0x50))
    pic, *_ = add_picture_fit(s, ROOT / "results" / "figures" / "cover_patch.jpg", 8.45, 1.10, 4.22, 4.22)
    pic.line.color.rgb = TEAL_L
    pic.line.width = Pt(0.75)
    add_text(s, "Massachusetts Buildings  ·  1 m 航拍影像", 8.45, 5.55, 4.3, 0.3, 9, False, TEAL_L)
    add_text(s, "毕业设计（论文）汇报", 0.8, 1.25, 6.6, 0.36, 13, True, TEAL_L)
    add_text(s, "基于深度学习的\n航拍图像建筑物自动提取", 0.8, 1.78, 7.3, 1.65, 35, True, WHITE, spacing=1.12)
    add_text(s, "多尺度特征融合与边界监督的初步研究", 0.82, 3.62, 7.1, 0.45, 17, False, TEAL_L)
    add_rect(s, 0.82, 4.35, 2.0, 0.035, ORANGE)
    add_text(s, "答辩人：________      指导教师：________", 0.82, 4.65, 7.0, 0.35, 12, False, RGBColor(0xC9, 0xD6, 0xE3))
    add_text(s, "2026 年 10 月", 0.82, 5.12, 4.0, 0.35, 11, False, RGBColor(0xC9, 0xD6, 0xE3))
    add_text(s, "Python · PyTorch · PyCharm · MATLAB", 0.82, 6.45, 7.0, 0.3, 9.5, False, TEAL_L)
    add_notes(s, "各位老师好，我的论文题目是《基于深度学习的航拍图像建筑物自动提取》。本次汇报重点介绍数据难点、方法设计、三组对照实验和失败案例分析。")

    # ---------------- 2. 目录 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "汇报提纲", "CONTENTS", 2)
    items = [
        ("01", "研究背景与挑战", "建筑提取的应用价值、四类核心困难与评价难点"),
        ("02", "数据与方法", "数据集难度量化、U-Net 基线、ASPP 与边界监督"),
        ("03", "实验与结果", "三组模型对照、边界/实例指标、分场景分析"),
        ("04", "失败案例分析", "密集粘连、阴影误检、小目标漏提、复杂背景"),
        ("05", "结论与展望", "本阶段结论、局限与下一阶段技术路线"),
    ]
    for i, (num, title, desc) in enumerate(items):
        y = 1.55 + i * 1.02
        add_text(s, num, 0.75, y, 0.9, 0.55, 25, True, TEAL if i != 4 else ORANGE)
        add_text(s, title, 1.75, y - 0.02, 4.0, 0.4, 16, True, NAVY)
        add_text(s, desc, 1.77, y + 0.38, 8.6, 0.32, 10.5, False, GRAY)
        if i < 4:
            add_rect(s, 1.75, y + 0.82, 10.2, 0.012, MID)
    add_notes(s, "汇报分为五个部分：背景挑战、数据方法、实验结果、失败案例和结论展望。")

    # ---------------- 3. 背景 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "研究背景：建筑提取是遥感智能解译的基础任务", "从像素到矢量，支撑城市规划与灾害评估", 3)
    cards = [
        ("城市规划", "建筑密度、容积率与功能区识别的基础数据", TEAL),
        ("灾害评估", "震后/洪涝房屋损毁范围的快速圈定", ORANGE),
        ("地图更新", "基础测绘与三维城市模型的高效更新", GREEN),
        ("违建监测", "变化检测与土地执法的线索发现", TEAL),
    ]
    for i, (t, b, c) in enumerate(cards):
        x = 0.55 + (i % 2) * 3.85
        y = 1.55 + (i // 2) * 2.05
        add_card(s, x, y, 3.55, 1.75, t, b, accent=c)
    add_rect(s, 8.55, 1.55, 4.2, 3.85, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    add_text(s, "18.7%", 8.85, 1.85, 3.6, 0.85, 38, True, TEAL_L)
    add_text(s, "测试集建筑像素占比", 8.88, 2.70, 3.4, 0.3, 11, False, WHITE)
    add_text(s, "建筑只占影像的很小一部分，却要在阴影、遮挡和密集排列中被逐像素识别。", 8.88, 3.15, 3.45, 1.35, 12, False, RGBColor(0xD5, 0xE1, 0xEA), spacing=1.25)
    add_text(s, "因此，通用语义分割模型不能直接照搬。", 8.88, 4.65, 3.5, 0.55, 10.5, True, ORANGE)
    add_text(s, "本阶段目标：建立可复现的 CPU 对照实验，量化四类困难，并判断改进模块的作用边界。",
             0.55, 5.85, 12.1, 0.8, 12.5, True, NAVY)
    add_notes(s, "建筑提取支撑城市规划、灾害评估、地图更新和违建监测。建筑像素只占约 18.7%，但要求逐像素准确，难度很高。")

    # ---------------- 4. 挑战 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "四类核心困难：从影像现象到评价指标", "数据集统计 + 失败模式 + 针对性指标", 4)
    challenges = [
        ("01", "光照变化", "阴影、曝光不足、屋顶反光", "边界对比度下降、纹理丢失", "按亮度分层评价", ORANGE),
        ("02", "遮挡与复杂背景", "树冠、道路、水体、油罐、码头", "建筑与背景混淆，误检增加", "ASPP 扩大上下文", TEAL),
        ("03", "尺度差异", "小屋顶仅几十像素，大厂房跨越百米", "小目标漏提、大建筑内部空洞", "多膨胀率分支 + 小目标召回", GREEN),
        ("04", "密集区域粘连", "联排住宅、厂房群屋顶紧邻", "相邻建筑被预测为同一连通域", "粘连率 + 实例分离", RGBColor(0x8A, 0x5A, 0x9E)),
    ]
    for i, (num, title, desc, impact, resp, color) in enumerate(challenges):
        x = 0.55 + (i % 2) * 6.15
        y = 1.55 + (i // 2) * 2.12
        add_rect(s, x, y, 5.85, 1.85, LIGHT, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        add_text(s, num, x + 0.18, y + 0.14, 0.55, 0.45, 18, True, color)
        add_text(s, title, x + 0.78, y + 0.14, 4.7, 0.35, 14.5, True, NAVY)
        add_text(s, desc, x + 0.22, y + 0.55, 5.4, 0.3, 10, False, GRAY)
        add_text(s, "影响：" + impact, x + 0.22, y + 0.92, 5.4, 0.3, 10, False, GRAY)
        add_text(s, "应对：" + resp, x + 0.22, y + 1.28, 5.4, 0.3, 10, True, color)
    add_text(s, "评价不能只看 IoU", 0.55, 6.05, 2.2, 0.3, 11, True, NAVY)
    pills = [("边界模糊 → Boundary IoU", TEAL), ("小目标漏提 → 小目标召回", GREEN), ("密集粘连 → 粘连率", ORANGE)]
    for i, (text, color) in enumerate(pills):
        x = 2.55 + i * 3.45
        add_rect(s, x, 6.00, 3.2, 0.42, color, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.3)
        add_text(s, text, x + 0.15, 6.06, 2.95, 0.28, 9.5, True, WHITE, PP_ALIGN.CENTER)
    add_notes(s, "我把困难拆成四类，并分别对应可量化的指标：边界用 Boundary IoU，小目标用实例召回，密集用粘连率。")

    # ---------------- 5. 研究现状 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "研究现状与本文定位", "语义分割、边界质量和实例可分性三条线索", 5)
    cols = [
        ("语义分割主线", TEAL,
         ["U-Net：编码—解码 + 跳跃连接，是小样本遥感分割的强基线", "DeepLabv3+ / ASPP：多膨胀率空洞卷积捕获多尺度上下文", "SegFormer / Swin：Transformer 编码器提升全局建模能力"],
         "局限：小目标与边界仍依赖高分辨率特征"),
        ("边界与实例", ORANGE,
         ["Boundary IoU：把评价从像素重叠推进到轮廓贴合", "边界损失 / 主动轮廓：显式约束轮廓质量", "分水岭、Mask R-CNN：从语义分割走向实例分离"],
         "局限：边界监督常与分割主任务争夺特征"),
        ("本文定位", GREEN,
         ["统一 CPU 预算下的三组对照：U-Net / +ASPP / +边界监督", "同时报告像素、边界、实例三类指标", "重点回答：改进模块到底改善了什么"],
         "不追求 SOTA，追求可复现的归因"),
    ]
    for i, (head, color, items, foot) in enumerate(cols):
        x = 0.55 + i * 4.18
        add_rect(s, x, 1.50, 3.95, 4.65, WHITE, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
        add_rect(s, x, 1.50, 3.95, 0.52, color, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
        add_rect(s, x, 1.80, 3.95, 0.22, color)
        add_text(s, head, x + 0.2, 1.58, 3.5, 0.32, 13, True, WHITE)
        add_bullets(s, items, x + 0.22, 2.22, 3.52, 2.9, size=9.8, spacing=1.18, bullet_color=color)
        add_rect(s, x + 0.2, 5.25, 3.55, 0.012, MID)
        add_text(s, foot, x + 0.2, 5.38, 3.55, 0.55, 9.2, True, color)
    add_text(s, "参考：Ronneberger et al., 2015；Chen et al., 2018；Cheng et al., 2021；Ji et al., 2019", 0.55, 6.45, 12.0, 0.3, 8.5, False, GRAY)
    add_notes(s, "相关工作主要有三条线：语义分割、边界评价与实例分离。本文的定位不是刷 SOTA，而是在统一预算下做归因实验。")

    # ---------------- 6. 技术路线 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "技术路线：一条可复现的对照实验链", "数据 → 基线 → 多尺度 → 边界监督 → 三级评价", 6)
    box_w, box_h = 3.05, 1.15
    row1 = [("01  数据与难度量化", 0.6), ("02  U-Net 基线", 4.25), ("03  ASPP 多尺度", 7.9)]
    row2 = [("04  边界辅助监督", 7.9), ("05  三级评价", 4.25), ("06  失败案例与改进", 0.6)]
    for text, x in row1:
        add_rect(s, x, 1.75, box_w, box_h, WHITE, line=TEAL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        add_text(s, text, x + 0.18, 2.00, box_w - 0.3, 0.45, 13, True, NAVY)
    for text, x in row2:
        add_rect(s, x, 4.25, box_w, box_h, WHITE, line=ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        add_text(s, text, x + 0.18, 4.50, box_w - 0.3, 0.45, 13, True, NAVY)
    # 箭头
    for x in [3.68, 7.33]:
        add_text(s, "→", x, 1.98, 0.5, 0.5, 22, True, TEAL, PP_ALIGN.CENTER)
    add_text(s, "↓", 9.0, 3.0, 0.5, 0.7, 22, True, ORANGE, PP_ALIGN.CENTER)
    for x in [3.68, 7.33]:
        add_text(s, "←", x, 4.48, 0.5, 0.5, 22, True, ORANGE, PP_ALIGN.CENTER)
    add_text(s, "统一预算：256×256 图块 · 900 张/轮 · 12 epochs · batch 8 · 阈值 0.5",
             0.6, 5.72, 12.1, 0.35, 10.5, True, GRAY, PP_ALIGN.CENTER)
    add_rect(s, 1.1, 6.15, 11.1, 0.55, LIGHT, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
    add_text(s, "核心判断：小样本 CPU 条件下，先保证可比性，再讨论模块增益。",
             1.2, 6.24, 10.9, 0.35, 11, True, TEAL, PP_ALIGN.CENTER)
    add_notes(s, "技术路线是一条闭环：数据量化、基线、ASPP、边界监督，最后用像素/边界/实例三级指标评价，并回到失败案例。")

    # ---------------- 7. 数据 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "数据：1 m 分辨率下的四类困难", "Massachusetts Buildings · 40 张训练影像 · 1325 个 256×256 图块", 7)
    stats = [("17,964", "测试集建筑数"), ("121 px²", "单栋面积中位数"), ("35.1%", "小于 100 px²"), ("61.1%", "图块超过 20 栋")]
    for i, (num, label) in enumerate(stats):
        x = 0.55 + i * 3.1
        add_rect(s, x, 1.15, 2.8, 0.72, WHITE, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
        add_text(s, num, x + 0.16, 1.19, 1.35, 0.35, 16, True, TEAL)
        add_text(s, label, x + 1.45, 1.25, 1.25, 0.28, 8.5, False, GRAY)
    add_picture_fit(s, ROOT / "results" / "figures" / "dataset_overview.png", 0.45, 2.02, 8.85, 4.85)
    add_text(s, "关键观察", 9.55, 2.05, 3.2, 0.35, 13, True, NAVY)
    add_bullets(s, ["小目标占主导：35.1% 的建筑小于 100 px²", "密集是常态：61.1% 的图块超过 20 栋", "光照偏暗：平均亮度仅 0.31", "分布偏移：测试集建筑覆盖率 18.7% > 训练集 13.6%"],
                9.55, 2.50, 3.2, 3.5, size=9.6, spacing=1.22)
    add_text(s, "→ 单一 IoU 会被大建筑主导，必须补充边界与实例指标。", 9.55, 6.05, 3.2, 0.65, 9.5, True, ORANGE)
    add_notes(s, "测试集有近 1.8 万栋建筑，35% 小于 100 像素，61% 的图块超过 20 栋，而且测试集比训练集更密集、建筑更大。")

    # ---------------- 8. 方法 U-Net + ASPP ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "方法一：U-Net 基线 + ASPP 多尺度上下文", "用同一编码器，逐项加入多尺度模块", 8)
    stages = [("输入\n256×256×3", 0.55), ("编码器\n16/32/64/128", 2.35), ("ASPP\nr=1/2/4/8", 4.15), ("解码器\n128/64/32/16", 5.95), ("输出\n建筑掩膜", 7.75)]
    for i, (text, x) in enumerate(stages):
        color = TEAL if i in (1, 3) else (ORANGE if i == 2 else NAVY)
        add_rect(s, x, 1.55, 1.5, 0.82, WHITE, line=color, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        add_text(s, text, x + 0.06, 1.66, 1.38, 0.62, 8.8, True, color, PP_ALIGN.CENTER)
        if i < 4:
            add_text(s, "→", x + 1.52, 1.70, 0.28, 0.4, 16, True, GRAY, PP_ALIGN.CENTER)
    # 跳跃连接
    add_rect(s, 2.35, 2.55, 5.1, 0.02, MID)
    add_text(s, "跳跃连接：保留高分辨率边界细节", 3.0, 2.62, 3.8, 0.28, 8.5, False, GRAY)
    # ASPP 展开
    add_rect(s, 4.05, 3.25, 3.8, 1.55, LIGHT, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    add_text(s, "ASPP 五个分支", 4.22, 3.34, 3.4, 0.3, 10.5, True, NAVY)
    for i, text in enumerate(["1×1", "r=2", "r=4", "r=8", "GAP"]):
        add_rect(s, 4.22 + i * 0.7, 3.78, 0.62, 0.5, TEAL if i < 4 else ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
        add_text(s, text, 4.25 + i * 0.7, 3.88, 0.56, 0.28, 8.5, True, WHITE, PP_ALIGN.CENTER)
    add_text(s, "拼接 → 1×1 压缩 → Dropout 0.1", 4.22, 4.40, 3.4, 0.26, 8.5, False, GRAY)
    # 右侧要点
    add_text(s, "为什么加 ASPP？", 9.45, 1.55, 3.35, 0.35, 13, True, NAVY)
    add_bullets(s, ["小建筑要局部纹理，大建筑要更大感受野", "多尺度上下文利用街道、院落结构区分相邻屋顶", "全局分支提供场景级先验，抑制孤立误检"],
                9.45, 2.02, 3.3, 2.5, size=9.5, spacing=1.25)
    add_rect(s, 9.45, 4.55, 3.3, 0.9, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, "参数量 1.96M → 4.20M", 9.63, 4.68, 2.95, 0.28, 10.5, True, TEAL_L)
    add_text(s, "CPU 训练 +55%", 9.63, 5.00, 2.95, 0.28, 10.5, True, TEAL_L)
    add_text(s, "代价：更多容量，也需要更多数据与训练预算。", 0.6, 5.45, 8.3, 0.5, 10.5, True, ORANGE)
    add_notes(s, "基线是 4 级 U-Net。ASPP 在瓶颈层并联 1×1、r=2/4/8 和全局池化五个分支，再压缩回原通道。参数量约翻倍，训练时间增加 55%。")

    # ---------------- 9. 边界监督 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "方法二：边界辅助监督，让轮廓进入共享特征", "不改变推理结构，只在训练时增加边界任务", 9)
    add_picture_fit(s, ROOT / "results" / "figures" / "edge_target_demo.png", 0.5, 1.45, 8.05, 2.9)
    add_rect(s, 8.85, 1.45, 4.0, 2.9, LIGHT, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    add_text(s, "边界目标", 9.05, 1.58, 3.6, 0.3, 11, True, NAVY)
    add_text(s, "B = Dilate(G) − Erode(G)", 9.05, 2.00, 3.6, 0.35, 12.5, True, TEAL, font=FONT_EN)
    add_text(s, "5×5 椭圆结构元，边界带总宽约 4 像素（约 4 m）", 9.05, 2.42, 3.55, 0.55, 9.2, False, GRAY)
    add_text(s, "总损失", 9.05, 3.05, 3.5, 0.28, 10, True, NAVY)
    add_text(s, "BCE + 0.5·Dice\n+ 0.35·BCE_edge", 9.05, 3.36, 3.5, 0.65, 11.5, True, ORANGE, font=FONT_EN)
    # 底部三卡
    points = [
        ("共享解码器", "边界头接在解码器第一层，梯度回传到共享特征，而不是只做后处理。", TEAL),
        ("类别不平衡", "边界像素稀少，使用正样本权重 4 的 BCE，避免模型倾向全预测背景。", ORANGE),
        ("零推理开销", "边界头只在训练时启用；推理只走分割头，不增加部署成本。", GREEN),
    ]
    for i, (title, body, color) in enumerate(points):
        add_card(s, 0.55 + i * 4.1, 4.75, 3.85, 1.55, title, body, accent=color, title_size=11.5, body_size=9.4)
    add_text(s, "关键：边界监督改变的是特征学习，不是推理后处理。", 0.6, 6.58, 12.0, 0.35, 10.5, True, NAVY)
    add_notes(s, "边界目标由膨胀减腐蚀得到。边界头只在训练时存在，推理时去掉，因此不增加部署成本。")

    # ---------------- 10. 实验设置 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "实验设置与评价体系", "统一预算下的三组对照 · 像素 / 边界 / 实例三级指标", 10)
    data = [
        ["项目", "设置"],
        ["数据", "Massachusetts · 40 张训练图 / 1325 图块"],
        ["输入", "256×256×3 · 训练集均值方差标准化"],
        ["增强", "翻转 / 旋转 / 亮度 / 对比度 / Gamma / 噪声"],
        ["模型", "U-Net · +ASPP · +ASPP+边界监督"],
        ["优化", "AdamW 3e-4 · Cosine · weight decay 1e-4"],
        ["训练", "12 epochs · batch 8 · 900 图块/轮"],
        ["阈值", "0.5（验证集仅 4 景，不调参）"],
        ["硬件", "AMD Ryzen 7 8845H · CPU 8 线程 · 无 GPU"],
    ]
    add_table(s, data, 0.55, 1.45, 6.7, [1.15, 5.55], row_h=0.43, font_size=9.5)
    metrics = [
        ("像素级", "IoU / F1 / Precision / Recall", "整体重叠度，但会被大面积建筑主导", TEAL),
        ("边界级", "Boundary IoU（2 px 带）", "衡量轮廓贴合，不只看内部是否填满", ORANGE),
        ("实例级", "小目标召回（<100 px²）/ 粘连率", "直接对应漏提与密集粘连", GREEN),
    ]
    for i, (tag, formula, desc, color) in enumerate(metrics):
        y = 1.45 + i * 1.55
        add_rect(s, 7.55, y, 5.2, 1.32, WHITE, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
        add_rect(s, 7.55, y, 0.08, 1.32, color)
        add_text(s, tag, 7.78, y + 0.12, 1.0, 0.3, 10.5, True, color)
        add_text(s, formula, 8.75, y + 0.10, 3.8, 0.3, 10.5, True, NAVY)
        add_text(s, desc, 7.78, y + 0.55, 4.7, 0.55, 9.3, False, GRAY)
    add_text(s, "评价原则：先看 IoU，再看边界与实例；不用测试集调阈值。", 0.55, 6.55, 12.0, 0.35, 10.5, True, TEAL)
    add_notes(s, "实验统一使用 256 输入、900 张图块、12 轮训练。评价同时看像素 IoU、Boundary IoU、小目标召回和粘连率。")

    # ---------------- 11. 总体结果 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "总体结果：没有免费午餐，但模块各有所长", "测试集 360 图块 · 17,964 栋建筑 · 阈值 0.5", 11)
    add_picture_fit(s, ROOT / "matlab_output" / "matlab_paper_results.png", 0.45, 1.18, 9.2, 4.0)
    kpis = [
        ("基线", "IoU 0.583\nF1 0.737", TEAL),
        ("ASPP", "粘连率 −10.8%\nIoU −0.77%", ORANGE),
        ("+边界", "边界 IoU 0.3703\n总体持平", GREEN),
    ]
    for i, (tag, val, color) in enumerate(kpis):
        y = 1.30 + i * 1.38
        add_rect(s, 9.95, y, 2.9, 1.15, WHITE, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        add_rect(s, 9.95, y, 0.07, 1.15, color)
        add_text(s, tag, 10.15, y + 0.12, 2.5, 0.28, 10, True, color)
        add_text(s, val, 10.15, y + 0.42, 2.5, 0.6, 10.5, True, NAVY, spacing=1.1)
    add_bullets(s, ["ASPP 提升召回与小目标，却带来更多假阳性；总体 IoU 反而下降 0.45 个百分点。", "边界监督把 Boundary IoU 提到 0.3703，平均单图 IoU 三组最佳，但总体 IoU 与基线持平。", "完整模型相对基线的 95% Bootstrap 区间包含 0，尚不能称为显著提升。"],
                0.6, 5.35, 12.1, 1.4, size=9.5, spacing=1.15)
    add_notes(s, "总体结果是一个权衡：ASPP 降粘连但降 IoU，边界监督提边界但总体持平。这说明单一模块无法同时解决所有误差。")

    # ---------------- 12. 训练与消融 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "训练过程与消融：边界监督收敛更快", "验证集仅 4 景，曲线存在波动，结论以测试集为准", 12)
    add_picture_fit(s, ROOT / "results" / "figures" / "training_curves.png", 0.45, 1.35, 8.55, 2.35)
    add_text(s, "读图", 9.3, 1.38, 3.4, 0.3, 12, True, NAVY)
    add_bullets(s, ["完整模型从第 3 轮起验证 IoU 领先，边界任务起到正则化作用", "ASPP 收敛更慢，第 12 轮才接近基线", "验证曲线在 0.47–0.53 波动，单轮最优点有偶然性"],
                9.3, 1.82, 3.5, 2.6, size=9.3, spacing=1.2)
    deltas = [("ASPP", "小目标召回 +1.3%", "粘连率 −10.8%", ORANGE),
              ("+边界监督", "Boundary IoU +1.4%", "平均单图 IoU 最佳", GREEN),
              ("总体", "IoU 0.5830", "与基线持平，未显著", TEAL)]
    for i, (tag, a, b, color) in enumerate(deltas):
        x = 0.55 + i * 4.1
        add_rect(s, x, 4.35, 3.8, 1.45, WHITE, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        add_rect(s, x, 4.35, 0.08, 1.45, color)
        add_text(s, tag, x + 0.22, 4.48, 3.3, 0.3, 11, True, color)
        add_text(s, a, x + 0.22, 4.90, 3.35, 0.3, 9.8, False, NAVY)
        add_text(s, b, x + 0.22, 5.25, 3.35, 0.3, 9.8, False, GRAY)
    add_text(s, "训练成本：基线 19.4 min；ASPP 30.1 min；完整模型 29.7 min（CPU）。", 0.55, 6.15, 12.0, 0.35, 9.5, False, GRAY)
    add_notes(s, "完整模型收敛更快，但验证集太小，不能过度解读。消融显示 ASPP 改善小目标与粘连，边界监督改善边界，但总体 IoU 没有显著提升。")

    # ---------------- 13. 分场景 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "分场景分析：低亮度与稀疏场景最受伤", "按光照、阴影、密度、尺度、背景复杂度、覆盖率分层", 13)
    add_picture_fit(s, ROOT / "results" / "figures" / "difficulty_breakdown.png", 0.45, 1.42, 9.05, 5.2)
    add_text(s, "三个稳定现象", 9.8, 1.45, 3.0, 0.3, 12, True, NAVY)
    add_bullets(s, ["低亮度组 IoU 仅 0.52–0.53，比中等亮度低约 5 个百分点", "低覆盖率组 IoU 0.44–0.47，少量误检被放大", "密集组 IoU 约 0.59，但粘连率约 0.20，实例错误被 IoU 掩盖"],
                9.8, 1.90, 3.0, 3.7, size=9.2, spacing=1.2)
    add_rect(s, 9.8, 5.55, 2.95, 0.95, ORANGE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, "结论：光照与实例可分性\n是下一阶段的主攻方向", 9.95, 5.68, 2.7, 0.7, 9.5, True, WHITE)
    add_notes(s, "分场景看，低亮度最稳定地拉低精度；密集场景 IoU 不低，但粘连率约 0.20，说明语义 IoU 掩盖了实例错误。")

    # ---------------- 14. 定性 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "定性对比：完整模型边界更整齐", "绿色为真值，蓝色为预测；误差图绿 TP / 红 FP / 蓝 FN", 14)
    add_picture_fit(s, ROOT / "results" / "figures" / "qualitative_comparison.png", 0.45, 1.12, 8.85, 5.95)
    add_text(s, "逐行观察", 9.5, 1.15, 3.3, 0.3, 12, True, NAVY)
    add_bullets(s, ["阴影/港区：水面被大量误检，厂房内部出现空洞", "密集区：屋顶大体覆盖，但相邻屋顶仍粘连", "小目标：背阴、树荫、道路夹缝中的屋顶漏提", "复杂背景：低对比度建筑边界被截断"],
                9.5, 1.60, 3.35, 3.7, size=9.2, spacing=1.2)
    add_rect(s, 9.5, 5.55, 3.2, 0.9, LIGHT, line=MID, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    add_text(s, "边界监督改善了轮廓，但无法替代\n针对性的负样本挖掘。", 9.65, 5.66, 2.95, 0.65, 9.3, True, GRAY)
    add_notes(s, "定性看，完整模型的轮廓更整齐，但阴影水面误检和密集粘连仍然存在，说明边界监督不能替代负样本挖掘和实例建模。")

    # ---------------- 15. 失败案例 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "失败案例：四类错误仍然存在", "完整模型 · 典型失败图块", 15)
    add_picture_fit(s, ROOT / "results" / "figures" / "failure_cases.png", 0.30, 1.34, 5.40, 5.62)
    causes = [
        ("01", "密集粘连", "联排屋顶颜色、纹理高度相似，语义损失无法区分相邻实例。", ORANGE),
        ("02", "阴影误检", "油罐、船坞、码头等圆形/矩形人造结构与屋顶外观接近。", TEAL),
        ("03", "小目标漏提", "面积 <100 px² 的屋顶经 4 次下采样后特征几乎消失。", GREEN),
        ("04", "边界不完整", "树冠遮挡、反光、与道路同色时，模型倾向保守截断。", RGBColor(0x8A, 0x5A, 0x9E)),
    ]
    for i, (num, title, body, color) in enumerate(causes):
        y = 1.20 + i * 1.32
        add_text(s, num, 6.10, y, 0.45, 0.35, 12, True, color)
        add_text(s, title, 6.60, y - 0.02, 2.2, 0.32, 11.5, True, NAVY)
        add_text(s, body, 6.60, y + 0.34, 6.1, 0.45, 8.8, False, GRAY)
    add_rect(s, 6.10, 6.55, 6.7, 0.5, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
    add_text(s, "→ 下一步：高分辨率数据 + 难负样本挖掘 + 分水岭/实例头", 6.22, 6.64, 6.45, 0.3, 9.5, True, TEAL_L)
    add_notes(s, "失败案例归纳为四类：密集粘连、阴影误检、小目标漏提、边界不完整。它们分别对应不同的改进方向。")

    # ---------------- 16. 结论 ----------------
    s = prs.slides.add_slide(blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    add_header(s, "结论与展望", "初步研究回答了“改进了什么”，也暴露了“还缺什么”", 16)
    add_rect(s, 0.55, 1.42, 12.15, 0.62, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
    add_text(s, "核心判断：当前瓶颈是数据分辨率与实例级建模，而不是网络深度不足。",
             0.8, 1.54, 11.7, 0.35, 11.5, True, TEAL_L, PP_ALIGN.CENTER)
    # 左结论
    add_text(s, "本阶段结论", 0.55, 2.25, 5.5, 0.35, 13, True, NAVY)
    conclusions = [
        "完成可复现 CPU 闭环：数据—训练—评价—可视化全部脚本化",
        "量化困难：35.1% 建筑 <100 px²，61.1% 图块 >20 栋，平均亮度 0.31",
        "ASPP：小目标召回 +1.3%、粘连率 −10.8%，但 IoU −0.77%",
        "边界监督：Boundary IoU +1.4%，总体 IoU 与基线持平",
    ]
    for i, text in enumerate(conclusions):
        y = 2.72 + i * 0.82
        add_text(s, f"0{i + 1}", 0.6, y, 0.4, 0.3, 10, True, TEAL)
        add_text(s, text, 1.05, y - 0.02, 5.35, 0.62, 9.5, False, GRAY, spacing=1.1)
    # 右展望
    add_text(s, "下一阶段（按优先级）", 7.0, 2.25, 5.5, 0.35, 13, True, NAVY)
    nexts = [
        ("换数据", "WHU / Inria 0.3 m，解决小目标信息不足"),
        ("换编码器", "ResNet / EfficientNet / SegFormer 预训练"),
        ("改损失", "边界加权、Boundary IoU 代理、可学习权重"),
        ("做实例", "距离变换 + 分水岭，或 Mask R-CNN"),
    ]
    for i, (tag, text) in enumerate(nexts):
        y = 2.72 + i * 0.82
        add_rect(s, 7.0, y - 0.02, 0.72, 0.42, ORANGE if i < 2 else GREEN, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
        add_text(s, tag, 7.05, y + 0.04, 0.65, 0.26, 8.2, True, WHITE, PP_ALIGN.CENTER)
        add_text(s, text, 7.88, y - 0.02, 4.6, 0.6, 9.5, False, GRAY, spacing=1.1)
    add_text(s, "本阶段的价值：给出可复现基线、可解释的模块归因，以及明确的下一步。",
             0.55, 6.40, 12.1, 0.45, 9.5, True, TEAL, PP_ALIGN.CENTER)
    add_notes(s, "结论是：ASPP 和边界监督各有作用，但没有显著提升总体 IoU。下一步优先换高分辨率数据和预训练编码器，再做实例分离。")

    # ---------------- 17. 致谢 ----------------
    s = prs.slides.add_slide(blank)
    bg = s.background.fill
    bg.solid(); bg.fore_color.rgb = NAVY
    add_rect(s, 0, 0, 0.16, 7.5, TEAL)
    add_rect(s, 0.8, 1.45, 2.2, 0.04, ORANGE)
    add_text(s, "感谢聆听", 0.8, 1.75, 8.0, 0.95, 42, True, WHITE)
    add_text(s, "敬请各位老师批评指正", 0.82, 2.85, 7.0, 0.5, 18, False, TEAL_L)
    add_text(s, "基于深度学习的航拍图像建筑物自动提取", 0.82, 4.3, 8.0, 0.4, 13, False, RGBColor(0xC9, 0xD6, 0xE3))
    add_text(s, "答辩人：________      2026 年 10 月", 0.82, 4.85, 6.0, 0.35, 11, False, RGBColor(0xC9, 0xD6, 0xE3))
    add_text(s, "代码与结果：PyCharm + Python · MATLAB 分析 · 全部图表可复现", 0.82, 6.5, 8.5, 0.3, 9, False, TEAL_L)
    add_picture_fit(s, ROOT / "results" / "figures" / "cover_patch.jpg", 9.0, 1.65, 3.4, 3.4)
    add_notes(s, "我的汇报到此结束，感谢各位老师，请批评指正。")

    prs.save(OUT)
    print("saved", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
