"""
Latent-Space Mandelbrot Explorer
─────────────────────────────────
Move the mouse to travel through a 2-D latent space.
The latent coordinates are zipped with a Mandelbrot-set grid to produce
a continuously morphing fractal view.

Left panel  : Mandelbrot set coloured by the current latent warp
Right panel : raw latent-space energy map
Bottom bar  : live zip(latent_array, mandelbrot_iterations) stream
"""

import numpy as np
import pygame
import sys
import colorsys

# ── settings ────────────────────────────────────────────────────────
W, H        = 1280, 720          # window size
PANEL_W     = W // 2             # each panel width
MAND_RES    = (PANEL_W, H - 80) # mandelbrot panel resolution
LAT_RES     = (PANEL_W, H - 80) # latent panel resolution
BAR_H       = 80                 # bottom bar height
MAX_ITER    = 80                 # mandelbrot max iterations
FPS         = 30

# ── mandelbrot computation ──────────────────────────────────────────
def mandelbrot_grid(cx_range, cy_range, res, max_iter):
    """Return iteration-count array for the given complex-plane window."""
    w, h = res
    cx = np.linspace(cx_range[0], cx_range[1], w)
    cy = np.linspace(cy_range[0], cy_range[1], h)
    C = cx[np.newaxis, :] + 1j * cy[:, np.newaxis]   # shape (h, w)
    Z = np.zeros_like(C)
    counts = np.zeros(C.shape, dtype=np.float64)
    mask = np.ones(C.shape, dtype=bool)
    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        diverged = mask & (np.abs(Z) > 2.0)
        # smooth colouring
        counts[diverged] = i + 1 - np.log2(np.log2(np.abs(Z[diverged]) + 1e-10))
        mask[diverged] = False
    return counts          # 0 → in-set, >0 → escaped

# ── latent helpers ──────────────────────────────────────────────────
def latent_energy(lx, ly, res):
    """Build a smooth 2-D energy field centred on (lx, ly)."""
    w, h = res
    xs = np.linspace(-2, 2, w)
    ys = np.linspace(-2, 2, h)
    X, Y = np.meshgrid(xs, ys)
    r2 = (X - lx) ** 2 + (Y - ly) ** 2
    energy = np.sin(5 * r2 + lx * 3) * np.cos(3 * X * ly) + np.exp(-r2)
    return energy

def hue_shift_palette(counts, hue_offset, max_iter):
    """Map iteration counts → RGB surface, shifting hue by hue_offset."""
    h, w = counts.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    in_set = counts == 0
    t = counts / max_iter            # 0-1
    hue = (t * 0.7 + hue_offset) % 1.0
    sat = np.where(in_set, 0.0, 0.85)
    val = np.where(in_set, 0.0, 0.6 + 0.4 * t)
    # vectorised hsv→rgb
    for yi in range(h):
        for xi in range(w):
            if in_set[yi, xi]:
                continue
            r, g, b = colorsys.hsv_to_rgb(hue[yi, xi], sat[yi, xi], val[yi, xi])
            rgb[yi, xi] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb

def fast_hsv_to_rgb(h, s, v):
    """Vectorised HSV→RGB (all inputs/outputs 0-1 numpy arrays)."""
    i = (h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    # build per-pixel
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return r, g, b

def make_mandelbrot_surface(counts, hue_offset, max_iter):
    """Counts → pygame Surface using vectorised HSV mapping."""
    h, w = counts.shape
    in_set = counts == 0
    t = np.clip(counts / max_iter, 0, 1)
    hue = (t * 0.7 + hue_offset) % 1.0
    sat = np.where(in_set, 0.0, 0.85)
    val = np.where(in_set, 0.0, 0.6 + 0.4 * t)
    r, g, b = fast_hsv_to_rgb(hue, sat, val)
    arr = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8)
    ], axis=-1)
    surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
    return surf

def make_latent_surface(energy):
    """Energy array → pygame Surface (blue-white heat map)."""
    e = energy.copy()
    e -= e.min()
    mx = e.max()
    if mx > 0:
        e /= mx
    r = (e * 80).astype(np.uint8)
    g = (e * 180).astype(np.uint8)
    b = (120 + e * 135).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
    return surf

def make_zip_bar(latent_vec, mand_vec, width, height):
    """Visualise zip(latent_array, mandelbrot_array) as a colour bar."""
    surf = pygame.Surface((width, height))
    surf.fill((15, 15, 25))
    n = min(len(latent_vec), len(mand_vec), width)
    for i, (lv, mv) in enumerate(zip(latent_vec[:n], mand_vec[:n])):
        # blend latent value and mandelbrot iteration into a colour
        h = (lv * 0.5 + mv * 0.003) % 1.0
        s = 0.8
        v = 0.4 + 0.6 * min(abs(lv), 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        col = (int(r * 255), int(g * 255), int(b * 255))
        pygame.draw.line(surf, col, (i, 4), (i, height - 4))
    return surf

# ── main ────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Latent-Space ⊗ Mandelbrot Explorer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    # base Mandelbrot (centred on the classic view)
    cx_range = (-2.5, 1.0)
    cy_range = (-1.2, 1.2)
    base_counts = mandelbrot_grid(cx_range, cy_range, MAND_RES, MAX_ITER)

    # pre-sample one row from mandelbrot for the zip bar
    mand_row = base_counts[base_counts.shape[0] // 2, :]

    prev_lx, prev_ly = None, None

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        mx, my = pygame.mouse.get_pos()

        # map mouse → latent coordinates  (-2 … 2)
        lx = (mx / W) * 4 - 2
        ly = (my / H) * 4 - 2

        # only recompute when mouse actually moved (cheap guard)
        if (lx, ly) != (prev_lx, prev_ly):
            prev_lx, prev_ly = lx, ly

            # ── left panel: hue-warped Mandelbrot ──
            hue_off = (lx + ly) * 0.15 + 0.5
            mand_surf = make_mandelbrot_surface(base_counts, hue_off, MAX_ITER)

            # ── right panel: latent energy map ──
            energy = latent_energy(lx, ly, LAT_RES)
            lat_surf = make_latent_surface(energy)

            # ── bottom bar: zip(latent_row, mandelbrot_row) ──
            lat_row = energy[energy.shape[0] // 2, :]
            zip_surf = make_zip_bar(lat_row, mand_row, W, BAR_H)

        # ── draw ──
        screen.fill((10, 10, 18))
        screen.blit(mand_surf, (0, 0))
        screen.blit(lat_surf, (PANEL_W, 0))
        screen.blit(zip_surf, (0, H - BAR_H))

        # labels
        lbl1 = font.render("Mandelbrot  (hue-warped by latent pos)", True, (200, 220, 255))
        lbl2 = font.render("Latent-Space Energy", True, (200, 220, 255))
        lbl3 = font.render(
            f"zip(latent, mandelbrot)   latent=({lx:+.2f}, {ly:+.2f})",
            True, (180, 200, 240)
        )
        screen.blit(lbl1, (8, H - BAR_H - 20))
        screen.blit(lbl2, (PANEL_W + 8, H - BAR_H - 20))
        screen.blit(lbl3, (8, H - BAR_H + 4))

        # crosshair on latent panel
        cx_px = int((lx + 2) / 4 * PANEL_W) + PANEL_W
        cy_px = int((ly + 2) / 4 * (H - BAR_H))
        pygame.draw.circle(screen, (255, 255, 255), (cx_px, cy_px), 6, 1)
        pygame.draw.line(screen, (255, 255, 255, 80), (cx_px - 10, cy_px), (cx_px + 10, cy_px))
        pygame.draw.line(screen, (255, 255, 255, 80), (cx_px, cy_px - 10), (cx_px, cy_px + 10))

        # divider
        pygame.draw.line(screen, (60, 60, 80), (PANEL_W, 0), (PANEL_W, H - BAR_H), 2)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
