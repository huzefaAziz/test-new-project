"""
"Single-Sample Prophet Inequalities" -- a piece for solo violin (and a second violin).
Chawla & Dang, arXiv:2610.03660

Run:   pip install pygame numpy
       python prophet_violin.py            (plays + shows a scrolling piano roll)
       python prophet_violin.py --wav      (also writes prophet_violin.wav)

How the paper becomes music
---------------------------
* Key: D minor (natural), ending on a D major chord (a Picardy third = "it works").
* The constants of the paper are the melodies. Each digit 0-9 is a scale degree
  (0..6 = D4..C5, 7..9 wrap into the next octave):
      1/(6*sqrt3) = 0.096225...   the main result
      0.5096                      random-order greedy (free disposal)
      1 - 1/e = 0.6321            divisible-item free disposal
      7/16 = 0.4375               the counterexample (Lemma 5.7)
      (1-1/e)/2 = 0.3161          concave divisible item
* Movements follow the paper:
   I   Abstract        - ONE sample: a single lone note, then the main theme.
   II  Reduction       - a canon: voice 1 = coordination sample, voice 2 = the real
                         buyers (same melody, delayed, a fifth lower: sample/real symmetry).
   III Free disposal   - phrases repeated while being "trimmed" (shorter and shorter),
                         like min{1, x} saturating.
   IV  Prophet 1/2     - a drone for the single item; notes below the threshold are
                         quiet and short, notes above it are loud and long
                         ("accept the first value that exceeds the threshold").
   V   Conclusion      - D major arpeggio, resolution.
"""
import sys
import numpy as np
import pygame

SR = 44100
BPM = 72
BEAT = 60.0 / BPM
SCALE = [0, 2, 3, 5, 7, 8, 10]          # D natural minor


def degree_to_midi(deg, base=62):          # base = D4
    octave, idx = divmod(deg, 7)
    return base + 12 * octave + SCALE[idx]


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# --------------------------------------------------------------------------- violin
def violin_note(freq, dur, vel=0.6):
    """Additive-synth violin: rich harmonics, formant bump, delayed vibrato, bow attack."""
    n = int((dur + 0.35) * SR)
    t = np.arange(n) / SR
    # vibrato that blooms after the attack
    depth = 0.0055 * np.clip((t - 0.18) / 0.35, 0, 1)
    inst_f = freq * (1 + depth * np.sin(2 * np.pi * 5.6 * t))
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    sig = np.zeros(n)
    for h in range(1, 20):
        fh = h * freq
        if fh > 9000:
            break
        amp = 1.0 / h ** 1.05
        amp *= 1 + 1.6 * np.exp(-((fh - 3000) / 1100) ** 2)    # bridge/body resonance
        amp *= 1 + 0.8 * np.exp(-((fh - 500) / 150) ** 2)      # air/body resonance
        sig += amp * np.sin(h * phase)
    # bow envelope: scratchy attack, slow swell, release
    att = 0.07
    env = np.minimum(t / att, 1.0)
    env *= 1 + 0.12 * np.sin(2 * np.pi * 0.7 * t)
    rel = np.where(t > dur, np.exp(-(t - dur) / 0.09), 1.0)
    env *= rel
    noise = np.random.randn(n) * 0.02 * np.exp(-t / 0.12)
    sig = (sig / 3.0 + noise) * env * vel
    return sig


# --------------------------------------------------------------------------- score
class Score:
    def __init__(self):
        self.notes = []       # (start_beat, midi, dur_beats, vel, voice)
        self.sections = []    # (start_beat, title)
        self.cursor = 0.0

    def section(self, title):
        self.sections.append((self.cursor, title))

    def note(self, midi, dur, vel=0.6, voice=0, at=None, advance=True):
        s = self.cursor if at is None else at
        self.notes.append((s, midi, dur, vel, voice))
        if advance and at is None:
            self.cursor += dur

    def digits(self, digs, rhythm, offset=0, vel=0.6, voice=0, at=None, vel_fn=None, dur_fn=None):
        """Play a digit string as a melody; returns total length in beats."""
        pos = self.cursor if at is None else at
        start = pos
        for i, d in enumerate(digs):
            dur = rhythm[i % len(rhythm)]
            v = vel
            if dur_fn:
                dur = dur_fn(int(d), dur)
            if vel_fn:
                v = vel_fn(int(d), vel)
            self.note(degree_to_midi(int(d) + offset), dur, v, voice, at=pos, advance=False)
            pos += dur
        if at is None:
            self.cursor = pos
        return pos - start


def compose():
    sc = Score()
    R = [1, 1, 2, 1, 0.5, 0.5, 1, 2]

    # I. Abstract -- one sample
    sc.section("I. Abstract  -  one single sample")
    sc.note(degree_to_midi(0), 4, 0.55)                       # the lone sample (D)
    sc.cursor += 1
    sc.digits("096225", [1, 1, 1, 1, 2, 4], vel=0.65)         # 1/(6*sqrt3)
    sc.cursor += 1

    # II. Reduction -- sample/real canon (coordination sample vs real buyers)
    sc.section("II.  Reduction  -  sample and real, in canon")
    start = sc.cursor
    melody = "5096" + "6321" + "4375" + "0962"
    length = sc.digits(melody, R, vel=0.6, voice=0, at=start)
    sc.digits(melody, R, offset=-4, vel=0.5, voice=1, at=start + 2)
    sc.cursor = start + length + 2 + 1

    # III. Free disposal -- trimmed, shorter and shorter
    sc.section("III.  Free disposal  -  min{1, x}: trimming")
    for dur in (2, 1, 0.5, 0.25):
        sc.digits("6321", [dur], vel=0.6 - 0.05 * (2 - dur if dur < 2 else 0), voice=0)
    sc.digits("6321", [0.25], vel=0.5)
    sc.cursor += 1

    # IV. Prophet inequality 1/2 -- threshold
    sc.section("IV.  Prophet inequality 1/2  -  accept above the threshold")
    start = sc.cursor
    sc.note(degree_to_midi(0, base=50), 8, 0.35, voice=1, at=start, advance=False)   # D3 drone
    sc.note(degree_to_midi(4, base=50), 8, 0.25, voice=1, at=start + 8, advance=False)  # A3
    thr = 4   # digits below the threshold are muted & brief, above are loud & long
    sc.digits("4375" "3161" "5096" "0962" "6321", [1], vel=0.6, voice=0, at=start,
              vel_fn=lambda d, v: 0.7 if d > thr else 0.22,
              dur_fn=lambda d, du: 2 if d > thr else 0.5)
    sc.cursor = start + 12
    sc.cursor += 1

    # V. Conclusion -- D major resolution (Picardy third)
    sc.section("V.  Conclusion  -  it works: D major")
    c = sc.cursor
    for i, m in enumerate([62, 66, 69, 74, 69, 66]):          # D F# A D A F#
        sc.note(m, 0.75, 0.6, at=c + i * 0.75, advance=False)
    c += 4.5
    for m, v in [(50, 0.4), (57, 0.35), (62, 0.5), (66, 0.5), (69, 0.55), (74, 0.65)]:
        sc.note(m, 6, v, voice=1 if m < 62 else 0, at=c, advance=False)
    sc.cursor = c + 6
    return sc


# --------------------------------------------------------------------------- render
def render(sc):
    total = (sc.cursor + 1) * BEAT + 1.0
    L = np.zeros(int(total * SR))
    Rr = np.zeros_like(L)
    pans = {0: 0.42, 1: 0.58}
    for start, midi, dur, vel, voice in sc.notes:
        s = violin_note(mtof(midi), dur * BEAT, vel)
        i = int(start * BEAT * SR)
        j = min(i + len(s), len(L))
        L[i:j] += s[: j - i] * (1 - pans[voice])
        Rr[i:j] += s[: j - i] * pans[voice]
    out = np.stack([L, Rr], axis=1)
    # small hall reverb
    wet = out.copy()
    for d, g in [(0.029, .30), (0.043, .25), (0.071, .20), (0.113, .15), (0.167, .10)]:
        k = int(d * SR)
        wet[k:] += g * out[:-k]
    out = wet
    out /= max(1e-9, np.abs(out).max()) / 0.85
    return (out * 32767).astype(np.int16)


def save_wav(data, path="prophet_violin.wav"):
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    print("wrote", path)


# --------------------------------------------------------------------------- display
def main():
    sc = compose()
    print("Synthesizing violin ...")
    audio = render(sc)
    if "--wav" in sys.argv:
        save_wav(audio)

    pygame.mixer.pre_init(SR, -16, 2, 1024)
    pygame.init()
    W, H = 1000, 520
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Single-Sample Prophet Inequalities - for violin")
    font = pygame.font.SysFont("serif", 22)
    small = pygame.font.SysFont("serif", 15)
    clock = pygame.time.Clock()

    sound = pygame.sndarray.make_sound(audio)
    sound.play()
    t0 = pygame.time.get_ticks()
    pps = 90                                # pixels per second
    lo, hi = 46, 80
    colors = {0: (214, 168, 90), 1: (120, 160, 210)}

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        now = (pygame.time.get_ticks() - t0) / 1000.0
        if now > len(audio) / SR:
            running = False
        screen.fill((22, 18, 16))
        x_play = W // 3
        # staff-like guide lines
        for m in range(lo, hi):
            y = H - 60 - (m - lo) * (H - 160) / (hi - lo)
            if m % 12 == 2:
                pygame.draw.line(screen, (50, 42, 38), (0, y), (W, y), 1)
        title = ""
        for sb, name in sc.sections:
            if now >= sb * BEAT:
                title = name
            x = x_play + (sb * BEAT - now) * pps
            if 0 <= x <= W:
                pygame.draw.line(screen, (90, 75, 60), (x, 70), (x, H - 30), 1)
        for start, midi, dur, vel, voice in sc.notes:
            x = x_play + (start * BEAT - now) * pps
            w = max(4, dur * BEAT * pps)
            if x > W or x + w < 0:
                continue
            y = H - 60 - (midi - lo) * (H - 160) / (hi - lo)
            active = start * BEAT <= now <= (start + dur) * BEAT
            col = colors[voice]
            if active:
                col = tuple(min(255, c + 60) for c in col)
            pygame.draw.rect(screen, col, (x, y - 6, w, 12), border_radius=6)
            if active:
                pygame.draw.rect(screen, (255, 245, 220), (x, y - 6, w, 12), 2, border_radius=6)
        pygame.draw.line(screen, (240, 230, 210), (x_play, 70), (x_play, H - 30), 2)
        screen.blit(font.render(title, True, (235, 220, 190)), (24, 20))
        screen.blit(small.render("Chawla & Dang - Single-Sample Prophet Inequalities (arXiv:2610.03660)",
                                 True, (150, 135, 115)), (24, H - 24))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
