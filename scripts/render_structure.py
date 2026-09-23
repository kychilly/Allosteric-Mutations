# Lowkey i messed up the code to create the figures, just know that theyre in data/PyMOL_figures, sorry ;-;

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

try:
    import pymol
    from pymol import cmd
except ImportError:
    print("[-] Error: PyMOL module not found. Please run this inside an environment with PyMOL installed.")
    sys.exit(1)


def find_line_pixel_center(temp_image_path):
    """Loads a rendered image containing a single blue dashed line, detects the blue pixels,
    and computes their exact 2D (x, y) pixel centroid."""
    img_bgr = cv2.imread(temp_image_path)
    if img_bgr is None:
        return None

    # PyMOL blue color filter (high blue, lower red/green)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define range for pure blue in HSV (PyMOL blue dashes)
    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find coordinates of all blue pixels belonging to the line
    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) == 0:
        # Fallback to direct BGR thresholding if HSV mask is empty
        b, g, r = cv2.split(img_bgr)
        mask = (b > 200) & (r < 100) & (g < 100)
        y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0:
        return None

    center_x = float(np.mean(x_indices))
    center_y = float(np.mean(y_indices))
    return (center_x, center_y)


def add_post_render_label(image_path, text, pixel_pos):
    """Draws text with a clean white outline centered directly on the target coordinate."""
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    x, y = pixel_pos
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text box directly on the detected 2D pixel center
    draw_x = x - (text_width / 2.0)
    draw_y = y - (text_height / 2.0)

    # Draw white text outline for high visibility against the protein ribbon
    stroke_w = 3
    for dx in range(-stroke_w, stroke_w + 1):
        for dy in range(-stroke_w, stroke_w + 1):
            if dx != 0 or dy != 0:
                draw.text((draw_x + dx, draw_y + dy), text, font=font, fill=(255, 255, 255, 255))

    # Draw primary dark blue text layer on top
    draw.text((draw_x, draw_y), text, font=font, fill=(0, 0, 200, 255))
    img.save(image_path)


def render_pymol_figures():
    print("[*] Generating publication-grade PyMOL figures...")

    pymol.finish_launching(['pymol', '-c'])

    clean_pdb_path = os.path.join("data", "1ZG4_clean.pdb")
    output_dir = os.path.join("data", "PyMOL_figures")
    os.makedirs(output_dir, exist_ok=True)

    detailed_image_path = os.path.join(output_dir, "pymol_allosteric_pathways.png")
    global_overview_path = os.path.join(output_dir, "global_boundary_overview.png")
    temp_dash_path = os.path.join(output_dir, "temp_dash.png")

    pymol.cmd.reinitialize()
    pymol.cmd.load(clean_pdb_path, "tem1")
    obj = "tem1"

    width, height = 2400, 1800
    pymol.cmd.viewport(width, height)

    # Rendering settings
    pymol.cmd.bg_color("black")  # Set to black matching your preferred style
    pymol.cmd.set("ray_shadows", "1")
    pymol.cmd.set("antialias", "3")
    pymol.cmd.set("specular", "0.3")
    pymol.cmd.set("cartoon_side_chain_helper", 1)

    # Base styling: Light gray cartoon ribbon
    pymol.cmd.hide("everything", obj)
    pymol.cmd.show("cartoon", obj)
    pymol.cmd.color("gray80", obj)

    # Active Site Cluster
    pymol.cmd.select("active_site", f"{obj} and resi 70+73+166")
    pymol.cmd.show("spheres", "active_site")
    pymol.cmd.color("hotpink", "active_site")

    candidate_bottlenecks = [68, 130, 172, 215, 235, 276]
    pymol.cmd.select("active_anchor", f"{obj} and resi 70 and name CA")

    valid_bottlenecks = []
    distance_records = []

    for resi in candidate_bottlenecks:
        bt_sel_name = f"bt_{resi}"
        pymol.cmd.select(bt_sel_name, f"{obj} and resi {resi} and name CA")

        model_bt = pymol.cmd.get_model(bt_sel_name)
        model_ac = pymol.cmd.get_model("active_anchor")

        if model_bt.atom and model_ac.atom:
            c1 = np.array(model_bt.atom[0].coord)
            c2 = np.array(model_ac.atom[0].coord)
            dist_val = np.linalg.norm(c1 - c2)

            # Filter long-range allosteric sites >= 15.0 Å
            if dist_val >= 15.0:
                valid_bottlenecks.append(resi)
                dist_name = f"dist_res_{resi}"
                pymol.cmd.distance(dist_name, bt_sel_name, "active_anchor")

                # Style dashed lines
                pymol.cmd.hide("labels", dist_name)
                pymol.cmd.color("blue", dist_name)
                pymol.cmd.set("dash_as_cylinders", 0, dist_name)
                pymol.cmd.set("dash_width", 4.0, dist_name)
                pymol.cmd.set("dash_gap", 0.30, dist_name)
                pymol.cmd.set("dash_radius", 0.08, dist_name)

                distance_records.append((resi, dist_val, dist_name))

    print(f"[*] Filtered long-range allosteric nodes (>= 15 Å): {valid_bottlenecks}")

    # Display selected long-range bottleneck residues as firebrick sticks
    if valid_bottlenecks:
        bottleneck_str = "+".join(map(str, valid_bottlenecks))
        pymol.cmd.select("bottlenecks", f"{obj} and resi {bottleneck_str}")
        pymol.cmd.show("sticks", "bottlenecks")
        pymol.cmd.color("firebrick", "bottlenecks")
        pymol.cmd.set("stick_radius", "0.4", "bottlenecks")
    else:
        pymol.cmd.select("bottlenecks", "none")

    # ==========================================
    # 1. GENERATE GLOBAL BOUNDARY OVERVIEW
    # ==========================================
    print("[*] Generating Global 15-20 Å Boundary Overview figure...")
    pymol.cmd.hide("dash", "all")
    pymol.cmd.zoom(obj, "1.8")
    pymol.cmd.turn("y", 20)
    pymol.cmd.turn("x", 10)

    pymol.cmd.ray(width, height)
    pymol.cmd.png(global_overview_path, dpi=300, ray=1)
    print(f"[+] Saved global overview to {global_overview_path}")

    # ==========================================
    # 2. GENERATE DETAILED ZOOMED NETWORK (Your Exact Script)
    # ==========================================
    print("[*] Generating detailed zoomed network figure...")
    pymol.cmd.select("functional_core", "active_site or bottlenecks")
    pymol.cmd.center("functional_core")
    pymol.cmd.zoom("functional_core", "3")
    pymol.cmd.turn("y", 35)
    pymol.cmd.turn("x", -15)

    # Ray-trace master image with all lines visible
    print("[*] Ray-tracing detailed master figure...")
    pymol.cmd.ray(width, height)
    pymol.cmd.png(detailed_image_path, dpi=300, ray=1)

    # Find exact 2D pixel coordinates for each line by isolating them temporarily
    final_label_data = []
    pymol.cmd.hide("dash", "all")

    for resi, dist_val, dist_name in distance_records:
        pymol.cmd.show("dash", dist_name)
        pymol.cmd.ray(width, height)
        pymol.cmd.png(temp_dash_path, dpi=300, ray=1)

        pixel_center = find_line_pixel_center(temp_dash_path)
        if pixel_center:
            print(f"[+] Residue {resi} ({dist_val:.1f} Å) 2D Pixel Center detected at: X={pixel_center[0]:.1f}, Y={pixel_center[1]:.1f}")
            final_label_data.append((f"{dist_val:.1f} Å", pixel_center))
        else:
            print(f"[-] Warning: Could not detect pixel center for residue {resi}")

        pymol.cmd.hide("dash", dist_name)

    # Restore all dashes for the final master image
    for _, _, dist_name in distance_records:
        pymol.cmd.show("dash", dist_name)

    # Clean up temp file
    if os.path.exists(temp_dash_path):
        os.remove(temp_dash_path)

    # Apply labels precisely on the detected 2D pixel centers
    print("[*] Applying post-render distance labels onto detected 2D centers...")
    for label_text, pixel_pos in final_label_data:
        add_post_render_label(detailed_image_path, label_text, pixel_pos)

    print(f"[+] Saved detailed figure to {detailed_image_path}")
    pymol.cmd.quit()


if __name__ == "__main__":
    render_pymol_figures()