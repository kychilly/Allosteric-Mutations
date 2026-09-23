# scripts/render_structure.py
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pymol
    from pymol import cmd
except ImportError:
    print("[-] Error: PyMOL module not found. Please run this inside an environment with PyMOL installed.")
    sys.exit(1)


def get_screen_coords(model_coord, img_width=2400, img_height=1800):
    """Map 3D world coordinate to 2D image pixel space (X, Y) using PyMOL view matrices."""
    view = cmd.get_view()
    rot = np.array(view[0:9]).reshape((3, 3))
    pos = np.array(view[9:12])
    origin = np.array(view[12:15])

    # Transform 3D coordinate through camera projection
    pt = model_coord - origin
    pt_cam = np.dot(rot, pt) + pos

    # Scale camera matrix field of view to pixel dimension
    fov = cmd.get("field_of_view")
    fov_rad = np.radians(float(fov) if fov is not None else 25.0)
    scale = (img_height / 2.0) / np.tan(fov_rad / 2.0)

    px = (img_width / 2.0) + (pt_cam[0] * scale / abs(pt_cam[2]))
    py = (img_height / 2.0) - (pt_cam[1] * scale / abs(pt_cam[2]))
    return float(px), float(py)


def add_post_render_label(image_path, text, pixel_pos):
    """Draws text with a clean white outline centered directly on the target coordinate."""
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    x, y = pixel_pos

    # Get bounding box to center text exactly on the midpoint
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text box directly on (x, y) so it sits right on the dashed line
    draw_x = x - (text_width / 2.0)
    draw_y = y - (text_height / 2.0)

    # Draw white text outline (stroke) for high visibility against the protein ribbon
    stroke_w = 3
    for dx in range(-stroke_w, stroke_w + 1):
        for dy in range(-stroke_w, stroke_w + 1):
            if dx != 0 or dy != 0:
                draw.text((draw_x + dx, draw_y + dy), text, font=font, fill=(255, 255, 255, 255))

    # Draw primary dark blue text layer on top
    draw.text((draw_x, draw_y), text, font=font, fill=(0, 0, 200, 255))

    img.save(image_path)


def render_pymol_figure():
    print("[*] Generating final publication-grade PyMOL render...")

    pymol.finish_launching(['pymol', '-c'])

    clean_pdb_path = os.path.join("data", "1ZG4_clean.pdb")
    eval_dir = os.path.join("data", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_image_path = os.path.join(eval_dir, "pymol_allosteric_pathways.png")

    pymol.cmd.reinitialize()
    pymol.cmd.load(clean_pdb_path, "tem1")
    obj = "tem1"

    # Set exact viewport resolution to match ray-tracing dimensions for precise projection mapping
    width, height = 2400, 1800
    pymol.cmd.viewport(width, height)

    # Set background and rendering settings
    pymol.cmd.bg_color("white")
    pymol.cmd.set("ray_shadows", "1")
    pymol.cmd.set("antialias", "3")
    pymol.cmd.set("specular", "0.3")
    pymol.cmd.set("cartoon_side_chain_helper", 1)

    # Base styling: Light gray cartoon ribbon
    pymol.cmd.hide("everything", obj)
    pymol.cmd.show("cartoon", obj)
    pymol.cmd.color("gray80", obj)

    # 1. Active Site Cluster (Serine 70 triad) -> Spheres
    pymol.cmd.select("active_site", f"{obj} and resi 70+73+166")
    pymol.cmd.show("spheres", "active_site")
    pymol.cmd.color("hotpink", "active_site")

    # 2. Allosteric Communication Bottlenecks -> Distinct Firebrick Sticks
    bottleneck_list = [68, 130, 235]
    bottleneck_str = "+".join(map(str, bottleneck_list))
    pymol.cmd.select("bottlenecks", f"{obj} and resi {bottleneck_str}")
    pymol.cmd.show("sticks", "bottlenecks")
    pymol.cmd.color("firebrick", "bottlenecks")
    pymol.cmd.set("stick_radius", "0.4", "bottlenecks")

    # 3. Tight Framing: Center and orient view FIRST so view matrices are finalized
    pymol.cmd.select("functional_core", "active_site or bottlenecks")
    pymol.cmd.center("functional_core")
    pymol.cmd.zoom("functional_core", "3")
    pymol.cmd.turn("y", 35)
    pymol.cmd.turn("x", -15)

    # 4. Multiple Dark Blue Dashed Distance Lines (Collecting midpoints after framing)
    pymol.cmd.select("active_anchor", f"{obj} and resi 70 and name CA")

    distance_data = []
    for resi in bottleneck_list:
        bt_sel_name = f"bt_{resi}"
        dist_name = f"dist_res_{resi}"

        pymol.cmd.select(bt_sel_name, f"{obj} and resi {resi} and name CA")
        pymol.cmd.distance(dist_name, bt_sel_name, "active_anchor")

        # Hide native PyMOL labels since we use post-render PIL overlay
        pymol.cmd.hide("labels", dist_name)
        pymol.cmd.color("blue", dist_name)
        pymol.cmd.set("dash_as_cylinders", 0, dist_name)
        pymol.cmd.set("dash_width", 4.0, dist_name)
        pymol.cmd.set("dash_gap", 0.30, dist_name)
        pymol.cmd.set("dash_radius", 0.08, dist_name)

        # Extract 3D coordinates to calculate midpoint for the label overlay
        model_bt = pymol.cmd.get_model(bt_sel_name)
        model_ac = pymol.cmd.get_model("active_anchor")
        if model_bt.atom and model_ac.atom:
            c1 = np.array(model_bt.atom[0].coord)
            c2 = np.array(model_ac.atom[0].coord)
            dist_val = np.linalg.norm(c1 - c2)
            midpoint = (c1 + c2) / 2.0
            distance_data.append((dist_val, midpoint))

    print("[*] Ray-tracing final figure...")
    pymol.cmd.ray(width, height)
    pymol.cmd.png(output_image_path, dpi=300, ray=1)

    # 5. Post-process image to overlay crisp distance text with white outlines at exact screen positions
    print("[*] Applying post-render distance labels...")
    for dist_val, midpoint in distance_data:
        screen_x, screen_y = get_screen_coords(midpoint, width, height)
        label_text = f"{dist_val:.1f} Å"
        add_post_render_label(output_image_path, label_text, (screen_x, screen_y))

    print(f"[+] Saved final figure to {output_image_path}")
    pymol.cmd.quit()


if __name__ == "__main__":
    render_pymol_figure()