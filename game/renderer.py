"""Text and image renderers (the image renderer needs only Pillow)."""
from .state import SIZE, TREES, NODES, NODE_IDX, BASES, cell
from . import rules

COL_BG = (245, 242, 232)
COL_GRID = (200, 195, 180)
COL_P = [(52, 120, 220), (214, 62, 62)]
COL_P_LIGHT = [(190, 215, 250), (250, 200, 200)]
COL_TREE = (60, 130, 70)
COL_NODE = (40, 190, 200)


def render_text(s):
    ch = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    for t in TREES:
        ch[t // SIZE][t % SIZE] = "T"
    for i, n in enumerate(NODES):
        o = s.post[i]
        ch[n // SIZE][n % SIZE] = "N" if o == -1 else ("n" if o == 0 else "m")
    for p in (0, 1):
        b = BASES[p]
        ch[b // SIZE][b % SIZE] = "B" if p == 0 else "R"
    for p in (0, 1):
        for u in s.units[p]:
            ch[u[0] // SIZE][u[0] % SIZE] = str(u[1]) if p == 0 else chr(ord("a") + u[1] - 1)
    rows = [" ".join(r) for r in ch]
    sc = rules.scores(s)
    head = (f"turn {s.turn:2d}  player {s.player}  ap {s.ap}   "
            f"res {s.res}  base_hp {s.bhp}  score {sc}")
    return head + "\n" + "\n".join(rows)


def render_image(s, last=None, title="", cell_px=72, panel_px=230, labels=("Blue", "Red")):
    """Return a PIL image of the position.  `last` = (player, action) shown in the panel."""
    from PIL import Image, ImageDraw, ImageFont
    from .actions import action_str
    W = SIZE * cell_px
    img = Image.new("RGB", (W + panel_px, W), COL_BG)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default(size=15)
        fs = ImageFont.load_default(size=12)
        fb = ImageFont.load_default(size=18)
    except TypeError:
        f = fs = fb = ImageFont.load_default()
    for r in range(SIZE):
        for c in range(SIZE):
            x0, y0 = c * cell_px, r * cell_px
            d.rectangle([x0, y0, x0 + cell_px, y0 + cell_px], outline=COL_GRID)
    for t in TREES:
        r, c = divmod(t, SIZE)
        cx, cy = c * cell_px + cell_px // 2, r * cell_px + cell_px // 2
        d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=COL_TREE)
    for i, n in enumerate(NODES):
        r, c = divmod(n, SIZE)
        x0, y0 = c * cell_px, r * cell_px
        cx, cy = x0 + cell_px // 2, y0 + cell_px // 2
        d.polygon([(cx, cy - 22), (cx + 22, cy), (cx, cy + 22), (cx - 22, cy)], fill=COL_NODE)
        d.text((x0 + 4, y0 + 2), f"{s.stock[i]}", fill=(10, 80, 90), font=fs)
        if s.post[i] != -1:
            d.rectangle([x0 + 3, y0 + 3, x0 + cell_px - 3, y0 + cell_px - 3],
                        outline=COL_P[s.post[i]], width=4)
    for p in (0, 1):
        r, c = divmod(BASES[p], SIZE)
        x0, y0 = c * cell_px, r * cell_px
        d.rectangle([x0 + 6, y0 + 6, x0 + cell_px - 6, y0 + cell_px - 6], fill=COL_P_LIGHT[p],
                    outline=COL_P[p], width=3)
        d.text((x0 + 10, y0 + cell_px - 22), f"HP{s.bhp[p]}", fill=COL_P[p], font=fs)
    for p in (0, 1):
        for u in s.units[p]:
            r, c = divmod(u[0], SIZE)
            cx, cy = c * cell_px + cell_px // 2, r * cell_px + cell_px // 2
            d.ellipse([cx - 17, cy - 17, cx + 17, cy + 17], fill=COL_P[p],
                      outline=(0, 0, 0) if not u[2] else (255, 215, 0), width=3)
            d.text((cx - 4, cy - 8), str(u[1]), fill=(255, 255, 255), font=f)
    x = W + 12
    y = 10
    d.text((x, y), title or "Strategy AI", fill=(20, 20, 20), font=fb)
    y += 34
    sc = rules.scores(s)
    for p in (0, 1):
        d.text((x, y), f"{labels[p]}{'  <- to move' if s.player == p and not s.over else ''}",
               fill=COL_P[p], font=f)
        y += 20
        d.text((x, y), f"res {s.res[p] if s.res[p] >= 0 else '?'}  units {len(s.units[p])}"
                       f"  posts {s.post.count(p)}", fill=(30, 30, 30), font=fs)
        y += 16
        d.text((x, y), f"base HP {s.bhp[p]}   score {sc[p]}", fill=(30, 30, 30), font=fs)
        y += 28
    d.text((x, y), f"turn {s.turn // 2 + 1}/{rules.MAX_TURNS}   AP {s.ap}", fill=(30, 30, 30), font=f)
    y += 28
    if last:
        d.text((x, y), f"{labels[last[0]]}: {action_str(last[1])}", fill=COL_P[last[0]], font=fs)
        y += 22
    if s.over:
        w = s.winner
        d.text((x, y), "DRAW" if w == -1 else f"{labels[w]} wins", fill=(20, 20, 20), font=fb)
    return img
