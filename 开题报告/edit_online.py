# -*- coding: utf-8 -*-
"""Switch the first guidance record from in-person to online (idempotent)."""
import docx

PATH = r"D:\All4DS\BISHE\开题报告\2027届开题报告-基于多任务深度学习的遥感影像建筑物矢量化提取.docx"
d = docx.Document(PATH)
cell = d.tables[0].rows[4].cells[0]

found_form = found_place = False
for par in cell.paragraphs:
    text = par.text.strip()
    if text.startswith("指导形式："):
        found_form = True
        if text != "指导形式：线上指导":
            par.runs[0].text = "指导形式：线上指导"
            for r in par.runs[1:]:
                r.text = ""
    elif text.startswith("指导地点："):
        found_place = True
        if text != "指导地点：线上（腾讯会议）":
            par.runs[0].text = "指导地点：线上（腾讯会议）"
            for r in par.runs[1:]:
                r.text = ""

assert found_form and found_place, (found_form, found_place)
d.save(PATH)

check = docx.Document(PATH)
for par in check.tables[0].rows[4].cells[0].paragraphs:
    if par.text.strip().startswith(("指导形式：", "指导地点：")):
        print(par.text.strip())
