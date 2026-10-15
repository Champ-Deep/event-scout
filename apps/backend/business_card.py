"""
Premium single-sided business card (PNG), rendered server-side with Pillow.

White / ivory stock with black type and champagne-gold accents, split layout,
3.5" × 2" at 600 dpi (2100 × 1200 px):
  • left  — a crisp black vCard QR (scan → "Add contact") framed by gold corner marks,
            with a letter-spaced "SCAN TO CONNECT" caption
  • right — name (serif, black), title (gold small caps), company (charcoal), a gold rule,
            then LinkedIn ID, phone and email with fine gold icons
A thin gold inner border, a gold hairline divider and a discreet monogram finish it.
"""

import io
import os
import re
from typing import Optional

import qrcode
from PIL import Image, ImageDraw, ImageFont

W, H = 2100, 1200
PAPER = (252, 251, 247)       # warm white stock
PAPER_2 = (246, 243, 234)     # ivory band behind the QR
INK = (17, 17, 17)            # near-black type
CHARCOAL = (74, 74, 74)
GOLD = (176, 141, 72)         # deep champagne gold (readable on white)
GOLD_LIGHT = (212, 184, 124)

_FONT_CANDIDATES = {
    "serif_bold": [r"C:\Windows\Fonts\georgiab.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    "serif": [r"C:\Windows\Fonts\georgia.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "sans": [r"C:\Windows\Fonts\segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "sans_semibold": [r"C:\Windows\Fonts\seguisb.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "sans_light": [r"C:\Windows\Fonts\segoeuil.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES.get(kind, []):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)  # Pillow ≥10.1 bundled scalable font


def _fit(draw, text: str, kind: str, size: int, max_w: int, min_size: int = 28):
    while size > min_size:
        f = _font(kind, size)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size -= 4
    return _font(kind, min_size)


def _ellipsize(draw, text: str, font, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text + "…"


def _spaced_width(draw, text, font, tracking):
    return sum(draw.textlength(ch, font=font) + tracking for ch in text) - tracking if text else 0


def _spaced(draw, xy, text: str, font, fill, tracking: int):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def linkedin_id(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"linkedin\.com/(in|company)/([^/?#]+)", url, re.I)
    return f"{m.group(1).lower()}/{m.group(2)}" if m else url.replace("https://", "").replace("www.", "")


def _icon(draw, cx: int, cy: int, kind: str):
    """Fine gold line icons (drawn, so no icon font is needed)."""
    r = 25
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=GOLD, width=3)
    if kind == "in":
        f = _font("sans_semibold", 26)
        tw = draw.textlength("in", font=f)
        draw.text((cx - tw / 2, cy - 18), "in", font=f, fill=GOLD)
    elif kind == "phone":
        draw.rounded_rectangle((cx - 8, cy - 14, cx + 8, cy + 14), radius=4, outline=GOLD, width=3)
        draw.line((cx - 3, cy + 8, cx + 3, cy + 8), fill=GOLD, width=3)
    elif kind == "mail":
        draw.rectangle((cx - 13, cy - 9, cx + 13, cy + 9), outline=GOLD, width=3)
        draw.line((cx - 13, cy - 9, cx, cy + 2, cx + 13, cy - 9), fill=GOLD, width=3)


def _corner_marks(draw, x0, y0, x1, y1, length=70, width=5):
    """Gold L-shaped corner brackets around the QR (a lighter, more premium frame than a box)."""
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line((cx, cy, cx + dx * length, cy), fill=GOLD, width=width)
        draw.line((cx, cy, cx, cy + dy * length), fill=GOLD, width=width)


def _initials(name: str) -> str:
    return "".join(p[0] for p in (name or "").split()[:2]).upper() or "•"


def render_business_card(name: str, title: Optional[str], company: Optional[str], linkedin: Optional[str],
                         phone: Optional[str], email: Optional[str], qr_payload: str) -> bytes:
    card = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(card)

    # ivory panel behind the QR (left half) + thin gold inner border around the whole card
    split = W // 2 - 10
    d.rectangle((0, 0, split, H), fill=PAPER_2)
    d.rectangle((44, 44, W - 44, H - 44), outline=GOLD_LIGHT, width=2)
    d.rectangle((58, 58, W - 58, H - 58), outline=GOLD, width=3)

    # --- left: QR framed by gold corner marks ---
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_Q, box_size=20, border=0)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    qr_side = 620
    qr_img = qr.make_image(fill_color=INK, back_color=PAPER_2).convert("RGB").resize((qr_side, qr_side), Image.NEAREST)
    qx = (split - qr_side) // 2 + 10
    qy = (H - qr_side) // 2 - 36
    card.paste(qr_img, (qx, qy))
    pad = 34
    _corner_marks(d, qx - pad, qy - pad, qx + qr_side + pad, qy + qr_side + pad)
    cap = _font("sans_semibold", 28)
    cap_text = "SCAN TO CONNECT"
    cw = _spaced_width(d, cap_text, cap, 10)
    _spaced(d, (qx + (qr_side - cw) / 2, qy + qr_side + pad + 42), cap_text, cap, GOLD, 10)

    # gold hairline divider with a small diamond
    d.line((split, 150, split, H - 150), fill=GOLD, width=3)
    my = H // 2
    d.polygon([(split, my - 14), (split + 14, my), (split, my + 14), (split - 14, my)], fill=GOLD)

    # --- right: contact hierarchy ---
    x = split + 110
    max_w = W - x - 130
    y = 230
    name_f = _fit(d, name, "serif_bold", 112, max_w, 60)
    d.text((x, y), _ellipsize(d, name, name_f, max_w), font=name_f, fill=INK)
    y += name_f.size + 40
    if title:
        t = title.upper()
        tf = _fit(d, t, "sans_semibold", 36, max_w - 80, 24)
        _spaced(d, (x, y), _ellipsize(d, t, tf, max_w - 80), tf, GOLD, 6)
        y += tf.size + 24
    if company:
        cf = _fit(d, company, "serif", 48, max_w, 30)
        d.text((x, y), _ellipsize(d, company, cf, max_w), font=cf, fill=CHARCOAL)
        y += cf.size + 18
    y = max(y + 34, 610)
    d.line((x, y, x + 150, y), fill=GOLD, width=4)
    y += 58

    row_f = _font("sans", 40)
    for kind, value in (("in", linkedin_id(linkedin)), ("phone", phone), ("mail", email)):
        if not value:
            continue
        _icon(d, x + 25, y + 25, kind)
        d.text((x + 78, y - 1), _ellipsize(d, value, row_f, max_w - 90), font=row_f, fill=INK)
        y += 88

    # discreet monogram (top-right) + brand mark (bottom-right)
    mono_f = _font("serif_bold", 40)
    mono = _initials(name)
    mr = 46
    mcx, mcy = W - 150, 160
    d.ellipse((mcx - mr, mcy - mr, mcx + mr, mcy + mr), outline=GOLD, width=3)
    mw = d.textlength(mono, font=mono_f)
    d.text((mcx - mw / 2, mcy - 26), mono, font=mono_f, fill=GOLD)
    bf = _font("sans_semibold", 22)
    bw = _spaced_width(d, "EVENTSCOUT", bf, 7)
    _spaced(d, (W - 130 - bw, H - 118), "EVENTSCOUT", bf, (150, 150, 150), 7)

    out = io.BytesIO()
    card.save(out, format="PNG", optimize=True, dpi=(600, 600))
    return out.getvalue()
