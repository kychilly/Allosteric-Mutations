# scripts/preprocess.py
import os
import yaml
from Bio.PDB import PDBList, PDBParser, MMCIFParser, PDBIO, Select


class CleanPDBSelect(Select):
    """
    Custom selector to filter out water molecules (H2O),
    heteroatoms, non-standard chains, and hydrogens (keeping heavy atoms only),
    retaining only protein chain A.
    """

    def accept_residue(self, residue):
        resname = residue.get_resname().strip()
        standard_amino_acids = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"
        ]
        return resname in standard_amino_acids

    def accept_chain(self, chain):
        return chain.id == 'A'

    def accept_atom(self, atom):
        """
        Keep only heavy atoms (exclude hydrogens).
        """
        # Safely check element property if available, otherwise fall back to atom name
        element = getattr(atom, 'element', None)
        if element:
            element = element.strip().upper()
            if element == 'H':
                return False

        # Fallback check based on atom name (e.g., names starting with 'H' like HD2, HA)
        name = atom.get_name().strip().upper()
        if name.startswith('H') and (len(name) == 1 or name[1].isdigit() or name[1] in "ABGDEZTH"):
            # Ensure it's not a heavy atom whose name happens to start with H (like HG, HE in some naming conventions,
            # though standard heavy atoms are Hg, Ho, etc. - sticking to strict H + digit or single H)
            if len(name) == 1 or name[1].isdigit():
                return False

        return True


def run_preprocessing():
    # Load configuration from root config folder
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdb_id = config["target_pdb"]
    print(f"[*] Downloading base PDB structure ({pdb_id}) from RCSB...")

    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir="data", file_format="mmCif")

    # Locate downloaded structure file dynamically
    cif_filename = None
    for root, dirs, files in os.walk("data"):
        for file in files:
            if pdb_id.lower() in file.lower() and file.endswith(('.cif', '.ent', '.pdb')):
                cif_filename = os.path.join(root, file)
                break
        if cif_filename:
            break

    if not cif_filename or not os.path.exists(cif_filename):
        raise FileNotFoundError(f"Could not locate downloaded structure file for {pdb_id}")

    print(f"[*] Parsing and cleaning structure from {cif_filename}...")

    # Use MMCIFParser if it's a .cif file, otherwise use PDBParser
    if cif_filename.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(pdb_id, cif_filename)

    clean_output_path = os.path.join("data", f"{pdb_id}_clean.pdb")

    io = PDBIO()
    io.set_structure(structure)
    io.save(clean_output_path, CleanPDBSelect())

    print(f"[+] Preprocessing complete! Clean structural baseline saved to: {clean_output_path}")


if __name__ == "__main__":
    run_preprocessing()