import os
import sys

try:
    import pymol
    from pymol import cmd
except ImportError:
    print("[-] Error: PyMOL module not found. Please run this inside an environment with PyMOL installed.")
    sys.exit(1)


def render_pymol_figure():
    print("[*] Initializing PyMOL rendering pipeline...")

    # Start PyMOL in headless mode (-c means command-line/no GUI window)
    pymol.finish_launching(['pymol', '-c'])

    clean_pdb_path = os.path.join("data", "1ZG4_clean.pdb")
    if not os.path.exists(clean_pdb_path):
        raise FileNotFoundError(f"[X] Clean PDB not found at {clean_pdb_path}. Run preprocessing first!")

    eval_dir = os.path.join("data", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    output_image_path = os.path.join(eval_dir, "pymol_allosteric_pathways.png")

    print(f"[*] Loading structure from {clean_pdb_path}...")
    pymol.cmd.reinitialize()
    pymol.cmd.load(clean_pdb_path, "tem1")

    # General Aesthetics & Lighting Configuration
    pymol.cmd.bg_color("white")
    pymol.cmd.set("ray_shadows", "1")
    pymol.cmd.set("antialias", "3")
    pymol.cmd.set("specular", "0.2")
    pymol.cmd.set("direct", "0.8")

    # Hide default representations and set base cartoon style
    pymol.cmd.hide("everything", "tem1")
    pymol.cmd.show("cartoon", "tem1")
    pymol.cmd.color("white", "tem1")

    # 1. Highlight Active Site (Serine 70 active-site cluster in TEM-1)
    pymol.cmd.select("active_site", "tem1 and resi 70+73+166")
    pymol.cmd.show("spheres", "active_site")
    pymol.cmd.color("yellow", "active_site")

    # 2. Highlight High-Centrality Allosteric Bottlenecks (identified residues 68, 130, 235)
    pymol.cmd.select("bottlenecks", "tem1 and resi 68+130+235")
    pymol.cmd.show("sticks", "bottlenecks")
    pymol.cmd.color("firebrick", "bottlenecks")

    # 3. Define and Highlight the 15-20 Å Allosteric Boundary Shell relative to Active Site
    pymol.cmd.select(
        "boundary_shell",
        "tem1 and not active_site and not bottlenecks and (byres (tem1 within 20 of active_site) and not (tem1 within 15 of active_site))"
    )
    pymol.cmd.show("sticks", "boundary_shell")
    pymol.cmd.color("palecyan", "boundary_shell")

    # 4. Background structural context styling for remainder of protein
    pymol.cmd.color("gray85", "tem1 and not active_site and not bottlenecks and not boundary_shell")

    # Final Camera Framing & Ray Tracing
    pymol.cmd.center("active_site")
    pymol.cmd.zoom("active_site", "15")

    print("[*] Rendering high-resolution ray-traced image (this may take a few seconds)...")
    pymol.cmd.ray(2400, 1800)
    pymol.cmd.png(output_image_path, dpi=300)

    print(f"[+] PyMOL rendering complete! High-res figure saved to {output_image_path}")

    # Safely close PyMOL session
    pymol.cmd.quit()


if __name__ == "__main__":
    render_pymol_figure()