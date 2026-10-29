"""Generate synthetic business cards with ground truth, plus phone-photo style variants.

Usage: python3 bench/make_cards.py  -> bench/cards/{id}.png, {id}_photo.jpg, {id}.json
shortcut: synthetic cards only, add real photographed cards to bench/cards with a matching .json when available.
"""
import json
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(7)
OUT = os.path.join(os.path.dirname(__file__), "cards")
os.makedirs(OUT, exist_ok=True)
F = "/System/Library/Fonts/Supplemental/"
FONTS = {
    "sans": (F + "Arial.ttf", F + "Arial Bold.ttf"),
    "serif": (F + "Georgia.ttf", F + "Georgia Bold.ttf"),
    "mono": (F + "Courier New.ttf", F + "Courier New Bold.ttf"),
    "narrow": (F + "Arial Narrow.ttf", F + "Arial Narrow Bold.ttf"),
}

CARDS = [
    dict(name="Priya Raghunathan", title="VP, Clinical Operations", company_name="Medanta Health Systems",
         email="priya.r@medanta-health.com", phone="+91 98450 12345", linkedin="linkedin.com/in/priyaraghunathan",
         website="www.medanta-health.com"),
    dict(name="Ahmed Al Mansouri", title="Procurement Director", company_name="Gulf Medical Supplies LLC",
         email="a.mansouri@gulfmedsupplies.ae", phone="+971 50 123 4567", linkedin="", website="gulfmedsupplies.ae"),
    dict(name="Dr. Sarah O'Connell", title="Chief Medical Officer", company_name="Northbridge Diagnostics",
         email="soconnell@northbridgedx.com", phone="+1 (617) 555-0142", linkedin="linkedin.com/in/sarahoconnell-md",
         website="northbridgedx.com"),
    dict(name="Kenji Watanabe", title="Head of Partnerships, APAC", company_name="Hikari Imaging Co.",
         email="k.watanabe@hikari-imaging.jp", phone="+81 3-5555-0190", linkedin="", website="hikari-imaging.jp"),
    dict(name="Maria Fernanda Lopez", title="Marketing Manager", company_name="BioVida Pharma",
         email="mflopez@biovida.mx", phone="+52 55 1234 5678", linkedin="linkedin.com/in/mariafernandalopez", website=""),
    dict(name="Rohan Mehta", title="Founder & CEO", company_name="CareLoop AI",
         email="rohan@careloop.ai", phone="+91 99000 11223", linkedin="linkedin.com/in/rohanmehta", website="careloop.ai"),
    dict(name="Elizabeth Chen-Hartman", title="Senior Director, Supply Chain", company_name="Apex Hospital Group",
         email="echenhartman@apexhospitals.org", phone="+1 312 555 0199", linkedin="", website="apexhospitals.org"),
    dict(name="Omar Farouk", title="Regional Sales Lead", company_name="Nile Surgical",
         email="omar.farouk@nilesurgical.com", phone="+20 100 555 7788", linkedin="linkedin.com/in/omarfarouk", website=""),
    dict(name="Ingrid Svensson", title="Product Manager", company_name="Nordlys MedTech AB",
         email="ingrid.svensson@nordlys.se", phone="+46 70 123 45 67", linkedin="", website="www.nordlys.se"),
    dict(name="Tariq Hussain", title="IT Director", company_name="Al Noor Hospitals",
         email="t.hussain@alnoorhospitals.com", phone="+971 4 555 0101", linkedin="linkedin.com/in/tariqhussain", website="alnoorhospitals.com"),
    dict(name="Ananya Krishnan", title="Business Development Executive", company_name="Lake B2B",
         email="ananya.k@lakeb2b.com", phone="+91 80 4567 8900", linkedin="", website="www.lakeb2b.com"),
    dict(name="Jean-Luc Moreau", title="Directeur Commercial", company_name="Santé Connect SAS",
         email="jl.moreau@santeconnect.fr", phone="+33 6 12 34 56 78", linkedin="linkedin.com/in/jeanlucmoreau", website="santeconnect.fr"),
]

STYLES = [
    dict(bg=(255, 255, 255), fg=(20, 20, 20), accent=(0, 82, 155), font="sans", layout="left"),
    dict(bg=(18, 24, 38), fg=(240, 240, 240), accent=(80, 200, 160), font="sans", layout="left"),
    dict(bg=(250, 247, 240), fg=(40, 30, 20), accent=(150, 40, 40), font="serif", layout="center"),
    dict(bg=(255, 255, 255), fg=(30, 30, 30), accent=(230, 120, 0), font="narrow", layout="split"),
    dict(bg=(30, 30, 30), fg=(235, 235, 235), accent=(255, 200, 0), font="mono", layout="left"),
    dict(bg=(235, 242, 250), fg=(20, 40, 70), accent=(20, 120, 200), font="serif", layout="split"),
]


def font(path, size):
    return ImageFont.truetype(path, size)


def render(card, style):
    W, H = 1050, 600  # 3.5in x 2in at 300dpi
    img = Image.new("RGB", (W, H), style["bg"])
    d = ImageDraw.Draw(img)
    reg, bold = FONTS[style["font"]]
    initials = "".join(w[0] for w in card["company_name"].split()[:2]).upper()
    # logo block
    if style["layout"] == "split":
        d.rectangle([0, 0, 330, H], fill=style["accent"])
        d.text((60, 250), initials, font=font(bold, 90), fill=(255, 255, 255))
        x0 = 380
    else:
        d.rectangle([60, 50, 160, 150], fill=style["accent"])
        d.text((75, 72), initials, font=font(bold, 48), fill=(255, 255, 255))
        x0 = 60
    y = 190 if style["layout"] != "split" else 90
    lines = [
        (card["name"], bold, 46),
        (card["title"], reg, 28),
        (card["company_name"], bold, 30),
    ]
    contact = [("E", card["email"]), ("T", card["phone"])]
    if card["website"]:
        contact.append(("W", card["website"]))
    if card["linkedin"]:
        contact.append(("in", card["linkedin"]))
    if style["layout"] == "center":
        for text, f, s in lines:
            w = d.textlength(text, font=font(f, s))
            d.text(((W - w) / 2, y), text, font=font(f, s), fill=style["fg"])
            y += s + 14
        y += 30
        for _, v in contact:
            w = d.textlength(v, font=font(reg, 24))
            d.text(((W - w) / 2, y), v, font=font(reg, 24), fill=style["fg"])
            y += 36
    else:
        for text, f, s in lines:
            d.text((x0, y), text, font=font(f, s), fill=style["fg"])
            y += s + 14
        y += 30
        d.line([x0, y, x0 + 300, y], fill=style["accent"], width=3)
        y += 20
        for k, v in contact:
            d.text((x0, y), f"{k}  {v}", font=font(reg, 24), fill=style["fg"])
            y += 36
    return img


def photo_variant(img, hard=False):
    """Simulate a phone capture: perspective, blur, uneven light, glare, JPEG."""
    W, H = img.size
    pad = 160
    canvas = Image.new("RGB", (W + 2 * pad, H + 2 * pad), (random.randint(60, 120),) * 3)
    canvas.paste(img, (pad, pad))
    cw, ch = canvas.size
    j = 60 if hard else 25
    src = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    dst = [(random.randint(0, j), random.randint(0, j)), (cw - random.randint(0, j), random.randint(0, j)),
           (cw - random.randint(0, j), ch - random.randint(0, j)), (random.randint(0, j), ch - random.randint(0, j))]
    coeffs = _persp(dst, src)
    out = canvas.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    out = out.rotate(random.uniform(-4, 4) if not hard else random.uniform(-9, 9), resample=Image.BICUBIC, fillcolor=(90, 90, 90))
    # light gradient + glare
    grad = Image.linear_gradient("L").resize((cw, ch)).rotate(random.choice([0, 90, 180]))
    shade = Image.merge("RGB", (grad, grad, grad)).point(lambda p: 170 + p // 3)
    out = Image.blend(out, Image.composite(out, Image.new("RGB", (cw, ch), (0, 0, 0)), shade.convert("L")), 0.5)
    if hard:
        glare = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(glare).ellipse([cw * 0.5, ch * 0.1, cw * 0.95, ch * 0.6], fill=140)
        glare = glare.filter(ImageFilter.GaussianBlur(60))
        out = Image.composite(Image.new("RGB", (cw, ch), (255, 255, 255)), out, glare)
    out = out.filter(ImageFilter.GaussianBlur(1.6 if hard else 0.7))
    out.thumbnail((1600, 1600), Image.LANCZOS)  # matches app's max_dim
    return out


def _persp(pa, pb):
    import numpy as np
    A = []
    for (x, y), (X, Y) in zip(pa, pb):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(A, dtype=float)
    B = np.array([c for p in pb for c in p], dtype=float)
    return tuple(np.linalg.solve(A, B))


if __name__ == "__main__":
    n = 0
    for i, card in enumerate(CARDS):
        style = STYLES[i % len(STYLES)]
        cid = f"c{i + 1:02d}"
        img = render(card, style)
        img.save(f"{OUT}/{cid}.png")
        photo_variant(img).save(f"{OUT}/{cid}_photo.jpg", quality=85)
        photo_variant(img, hard=True).save(f"{OUT}/{cid}_hard.jpg", quality=70)
        json.dump(card, open(f"{OUT}/{cid}.json", "w"), indent=1)
        n += 3
    print(f"wrote {n} images for {len(CARDS)} cards to {OUT}")
