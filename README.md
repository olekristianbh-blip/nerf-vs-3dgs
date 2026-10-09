# NeRF vs 3D Gaussian Splatting — thesis companion

A small static webpage that compares **Neural Radiance Fields** and **3D Gaussian Splatting** side by side:

| Idea | NeRF (left) | 3DGS (right) |
|------|-------------|--------------|
| Representation | Continuous field queried at *(x, y, z)* | Explicit anisotropic Gaussians |
| Rendering | Ray march + volume integration | Rasterise / “splat” ellipsoids |
| View dependence | Direction *(θ, φ)* into the field | Spherical-harmonic-style appearance |
| Speed | Slow network optimisation (didactic bar) | Real-time novel views once fitted |

Guided steps walk through those four ideas; the last step is a free compare mode with a **shared camera**.

All geometry is a **didactic fake** — not a trained NeRF or a real Gaussian optimiser. Subsea / underwater optics are out of scope for this page.

## Open locally

The page loads Three.js from a CDN, so use a tiny static server (opening `index.html` as a `file://` URL can block ES modules).

From this folder:

```bash
npx --yes serve .
```

Or:

```bash
python -m http.server 8080
```

Then open the URL the command prints (typically `http://localhost:3000` or `http://localhost:8080`).

## How to read it

1. Walk the five steps with **Next** / **Back**, the dots, or the left/right arrow keys.
2. Drag in either panel to orbit — both views stay locked together.
3. On view-dependence and free-compare steps, move **θ** and **φ** to see colour change without moving the geometry.

## Sharing later (QR code)

This folder is a plain static site. For a public URL (and a QR code to the thesis poster):

1. Push the folder to a GitHub repository.
2. Enable **GitHub Pages** (deploy from the repo root or `/docs`).
3. Use the resulting `https://…github.io/…` URL in your QR code.

No build step is required.

## Files

- `index.html` — layout, step chrome, HUD
- `styles.css` — thesis-style page + split viewport
- `main.js` — dual Three.js scenes, shared orbit, didactic animations

## Related

Sibling companion: `../nerf-5d-viz` (NeRF as a 5D function).
