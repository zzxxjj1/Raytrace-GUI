# Interactive Ray Tracer

An interactive Python ray tracer with a Tkinter-based scene editor. The application renders a 512 × 512 scene of spheres and lets you adjust scene, camera, light, and material parameters before rendering again.

## Features

- Ray-sphere intersection and nearest-hit selection
- Ambient, diffuse, and specular lighting
- Shadow checks and recursive reflections
- Editable sphere position, radius, color, and reflectivity
- Editable camera position, light position, and material coefficients
- Multi-process rendering with shared image memory
- Rendering progress feedback in the GUI

## Requirements

- Python 3
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- Tkinter (included with many Python installations)

On Linux, you may need to install your distribution's Tk package first, such as `python3-tk`.

## Installation

```bash
git clone https://github.com/zzxxjj1/Raytrace-GUI.git
cd Raytrace-GUI

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pillow
```

## Run the Application

```bash
python raytrace_v4.py
```

The application opens a window and renders the default scene. Click **Render** after changing a setting to generate an updated image.

## Scene Controls

| Control | Description |
| --- | --- |
| **Edit Sphere 1–3** | Change a sphere's center coordinates, radius, RGB color values, and reflectivity. |
| **Edit Light and Material Properties** | Set the light position and the ambient, diffuse, and specular coefficients. |
| **Edit Camera Properties** | Change the camera position used to cast rays into the scene. |
| **Render** | Re-render the current scene using four worker processes. |

## How It Works

For every pixel, the renderer casts a ray from the camera into the scene and finds the closest sphere intersection. It computes local lighting from the surface normal and light direction, checks whether the point is shadowed, and blends in reflected color for reflective objects. Rendering work is divided into horizontal row ranges and processed in parallel through Python's `multiprocessing` module.

## Project Structure

```text
.
├── raytrace_v4.py   # Ray tracing logic, multiprocessing renderer, and Tkinter UI
└── README.md        # Project documentation
```

## Technologies

- Python
- NumPy
- Pillow
- Tkinter
- Python multiprocessing
