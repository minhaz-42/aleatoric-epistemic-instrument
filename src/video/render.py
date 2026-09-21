"""Rendering a scene's prefix to actual frames.

The instrument does not need pixels: every closed-form quantity is computed from the cue vector
directly, and the validation runs never call this module. It exists so the VLM arm is a real
experiment rather than a description of one -- a vision-language model has to be shown a video, and
a world that can only be described in text would test reading comprehension instead of perception.

The contract this module has to honour is narrow and absolute: **a cue that the mask hides must not
be recoverable from the pixels.** Every exact quantity downstream assumes the forecaster saw z_m and
nothing more, so a renderer that leaked a hidden cue would not merely add noise, it would falsify
the aleatoric floor that every result is measured against. Each cue is therefore hidden by removing
the evidence for it rather than by drawing it differently:

  heading   hidden -> the agent is drawn centred and rotationally symmetric, with no nose marker.
  speed     hidden -> no motion trail, and the agent holds position across the prefix.
  signal    hidden -> the signal head is replaced by an opaque occluder at the same location.
  blocker   hidden -> both branch mouths are covered by occluders, so absence is not informative.

The last one is the subtle case. Covering only the branch that happens to hold the blocker would
leak the cue through which side got covered, and drawing nothing at all would leak it through the
absence of an obstruction. Covering both branches unconditionally is the only option that reveals
nothing, and `test_instrument.py` asserts it by checking that the rendered frames of two scenes
differing only in a hidden cue are bitwise identical.
"""
import numpy as np

H = W = 256
ROAD_LO, ROAD_HI = 104, 152

_BG = np.array([0.42, 0.46, 0.40])
_ROAD = np.array([0.24, 0.24, 0.26])
_AGENT = np.array([0.95, 0.78, 0.20])
_TRAIL = np.array([0.62, 0.50, 0.16])
_OCCLUDER = np.array([0.14, 0.15, 0.14])
_SIGNAL = {-1.0: np.array([0.85, 0.15, 0.12]),
           0.0: np.array([0.55, 0.55, 0.55]),
           1.0: np.array([0.20, 0.80, 0.30])}
_BLOCK = np.array([0.60, 0.25, 0.15])


def _rect(img, y0, y1, x0, x1, colour, alpha=1.0):
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if y1 <= y0 or x1 <= x0:
        return
    img[y0:y1, x0:x1] = (1 - alpha) * img[y0:y1, x0:x1] + alpha * colour


def _disc(img, cy, cx, r, colour, alpha=1.0):
    yy, xx = np.ogrid[:H, :W]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[m] = (1 - alpha) * img[m] + alpha * colour


def _base_frame():
    img = np.tile(_BG, (H, W, 1)).astype(np.float64)
    _rect(img, 0, H, ROAD_LO, ROAD_HI, _ROAD)          # vertical road
    _rect(img, ROAD_LO, ROAD_HI, 0, W, _ROAD)          # horizontal road
    return img


def render_prefix(scene, world, n_frames: int = 8) -> np.ndarray:
    """(T, H, W, 3) float32 frames in [0, 1] for one scene's prefix.

    Reads `scene.z` only through `scene.mask`: a hidden cue never reaches the canvas.
    """
    names = [c.name for c in world.spec.cues]
    z = {n: (float(scene.z[i]) if scene.mask[i] else None) for i, n in enumerate(names)}

    heading, speed = z.get("heading"), z.get("speed")
    signal, blocker = z.get("signal"), z.get("blocker")

    # Approach geometry. With speed hidden the agent holds station, so no velocity is readable.
    y_start, travel = 240.0, 78.0
    frames = []
    for t in range(n_frames):
        u = t / max(1, n_frames - 1)
        if speed is None:
            cy = y_start                      # no displacement at all: nothing to measure
        else:
            cy = y_start - travel * u * (0.45 + 0.45 * (speed + 1) / 2)
        cx = 128.0 if heading is None else 128.0 + 13.0 * heading * (0.35 + 0.65 * u)

        img = _base_frame()

        # Signal head at the junction's far corner, or an occluder in its place.
        if signal is None:
            _rect(img, 86, 102, 160, 176, _OCCLUDER)
        else:
            _disc(img, 94, 168, 7, _SIGNAL[signal])

        # Branch mouths. Both are covered whenever the blocker is hidden, so neither the side nor
        # the absence of an obstruction is recoverable.
        if blocker is None:
            _rect(img, 112, 144, 62, 94, _OCCLUDER)
            _rect(img, 112, 144, 162, 194, _OCCLUDER)
        elif blocker < 0:
            _rect(img, 112, 144, 62, 94, _BLOCK)
        elif blocker > 0:
            _rect(img, 112, 144, 162, 194, _BLOCK)

        # Motion trail, only when speed is observable.
        if speed is not None:
            for k in range(1, 4):
                tu = max(0.0, u - 0.11 * k)
                ty = y_start - travel * tu * (0.45 + 0.45 * (speed + 1) / 2)
                tx = 128.0 if heading is None else 128.0 + 13.0 * heading * (0.35 + 0.65 * tu)
                _disc(img, ty, tx, 6, _TRAIL, alpha=0.75 / k)

        _disc(img, cy, cx, 11, _AGENT)
        # Nose marker: shows orientation, and is omitted entirely when heading is hidden.
        if heading is not None:
            _disc(img, cy - 8, cx + 7.0 * heading, 4, _AGENT * 0.55)

        frames.append(img)
    return np.clip(np.stack(frames), 0.0, 1.0).astype(np.float32)


def contact_sheet(frames: np.ndarray, cols: int = 4, pad: int = 4) -> np.ndarray:
    """Tile a prefix into one image.

    The VLM arm sends this rather than N separate images: it costs a fraction of the tokens, and it
    keeps the temporal order explicit in a single raster instead of relying on attachment order.
    """
    T = len(frames)
    rows = int(np.ceil(T / cols))
    h, w = frames.shape[1:3]
    sheet = np.ones((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), np.float32)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = f
    return sheet


def save_png(img: np.ndarray, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    mpimg.imsave(path, np.clip(img, 0, 1))
    return path
