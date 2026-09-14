"""TrueType outlines with explicit charset provenance; never silent fallback."""

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont


def save_ttf(glyphs, path, family="Personal Handwriting"):
    if not glyphs:
        raise ValueError("没有可导出的字形")
    chars = sorted(glyphs)
    names = {c: f"uni{ord(c):04X}" for c in chars}
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "space"] + [names[c] for c in chars if c != " "])
    cmap = {ord(c): names[c] for c in chars if c != " "}
    cmap[32] = "space"
    fb.setupCharacterMap(cmap)
    pen = TTGlyphPen(None)
    pen.moveTo((80, 0))
    pen.lineTo((80, 700))
    pen.lineTo((620, 700))
    pen.lineTo((620, 0))
    pen.closePath()
    gs = {".notdef": pen.glyph(), "space": TTGlyphPen(None).glyph()}
    metrics = {".notdef": (750, 80), "space": (500, 0)}
    for c in chars:
        if c == " ":
            continue
        item = glyphs[c]
        pen = TTGlyphPen(None)
        # Invert y: reverses all windings together, retaining counters/holes.
        for ring in item["polygons"]:
            if len(ring) < 3:
                continue
            p = [(round(x), round(800 - y)) for x, y in ring]
            pen.moveTo(p[0])
            for q in p[1:]:
                pen.lineTo(q)
            pen.closePath()
        glyph = pen.glyph()
        gs[names[c]] = glyph
        xs = [x for r in item["polygons"] for x, y in r]
        metrics[names[c]] = (round(item.get("advance", 1000)), round(min(xs)) if xs else 0)
    fb.setupGlyf(gs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=1000, descent=-300)
    fb.setupNameTable(
        {
            "familyName": family,
            "styleName": "Regular",
            "uniqueFontIdentifier": family + " 1.2",
            "fullName": family,
            "psName": "PersonalHandwriting-Regular",
            "version": "Version 1.2",
        }
    )
    fb.setupOS2(sTypoAscender=1000, sTypoDescender=-300, usWinAscent=1100, usWinDescent=350)
    fb.setupPost()
    fb.setupMaxp()
    fb.save(str(path))
    with TTFont(path) as font:
        if set(font.getBestCmap()) != set(cmap):
            raise RuntimeError("TTF cmap 验证失败")
