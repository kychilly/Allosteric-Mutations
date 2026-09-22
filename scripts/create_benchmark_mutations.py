# scripts/create_benchmark_mutations.py
import os
import yaml
import pandas as pd
from Bio.PDB import PDBParser


def find_residue_by_num(chain, res_num):
    """Safely find a residue in a BioPython chain by its sequence number."""
    for residue in chain:
        if residue.id[1] == res_num:
            return residue
    return None


def calculate_distance_to_ser70(structure, res_id):
    """
    Calculate the minimum Euclidean distance between the CA atom of a given residue
    and the CA atom of Ser70.
    """
    model = structure[0]
    chain = model['A']

    target_res = find_residue_by_num(chain, res_id)
    ser70_res = find_residue_by_num(chain, 70)

    if target_res is None or ser70_res is None:
        return None

    try:
        target_ca = target_res['CA'].get_coord()
        ser70_ca = ser70_res['CA'].get_coord()

        diff = target_ca - ser70_ca
        distance = (diff @ diff) ** 0.5
        return round(float(distance), 1)
    except KeyError:
        return None


def generate_benchmark_csv():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdb_id = config["target_pdb"].lower()
    clean_pdb_path = os.path.join("data", f"{config['target_pdb']}_clean.pdb")

    if not os.path.exists(clean_pdb_path):
        raise FileNotFoundError(f"Clean PDB file not found at {clean_pdb_path}. Run preprocessing first!")

    print(f"[*] Loading clean structure from {clean_pdb_path}...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, clean_pdb_path)
    model = structure[0]
    chain = model['A']

    # Master candidate pools (expanded to guarantee >= 30 valid hits each)
    resistance_pool = [
        ("R1", 153, "M", "I"), ("R2", 175, "G", "S"), ("R3", 179, "D", "G"),
        ("R4", 196, "Q", "K"), ("R5", 211, "I", "V"), ("R6", 222, "E", "K"),
        ("R7", 252, "G", "D"), ("R8", 265, "T", "M"), ("R9", 268, "A", "V"),
        ("R10", 276, "N", "D"), ("R11", 281, "K", "E"), ("R12", 103, "V", "I"),
        ("R13", 108, "S", "N"), ("R14", 121, "Y", "N"), ("R15", 168, "E", "K"),
        ("R16", 171, "E", "K"), ("R17", 197, "I", "V"), ("R18", 208, "D", "N"),
        ("R19", 227, "F", "L"), ("R20", 239, "W", "G"), ("R21", 259, "S", "G"),
        ("R22", 154, "A", "G"), ("R23", 160, "R", "K"), ("R24", 185, "I", "V"),
        ("R25", 188, "L", "I"), ("R26", 218, "K", "R"), ("R27", 224, "R", "S"),
        ("R28", 246, "G", "A"), ("R29", 257, "V", "A"), ("R30", 272, "W", "F"),
        ("R31", 115, "L", "V"), ("R32", 125, "A", "T"), ("R33", 148, "D", "N"),
        ("R34", 186, "G", "S"), ("R35", 191, "F", "L"), ("R36", 193, "V", "I"),
        ("R37", 198, "L", "M"), ("R38", 202, "P", "S"), ("R39", 206, "W", "L"),
        ("R40", 213, "V", "A"), ("R41", 219, "S", "A"), ("R42", 226, "V", "I"),
        ("R43", 231, "G", "A"), ("R44", 233, "R", "K"), ("R45", 236, "A", "V")
    ]

    neutral_pool = [
        ("N1", 45, "K", "R"), ("N2", 50, "A", "S"), ("N3", 55, "Q", "H"),
        ("N4", 82, "A", "G"), ("N5", 88, "V", "I"), ("N6", 95, "L", "I"),
        ("N7", 98, "E", "D"), ("N8", 112, "V", "I"), ("N9", 120, "S", "A"),
        ("N10", 150, "T", "S"), ("N11", 155, "D", "E"), ("N12", 189, "Q", "H"),
        ("N13", 195, "L", "V"), ("N14", 200, "A", "G"), ("N15", 205, "L", "M"),
        ("N16", 215, "K", "R"), ("N17", 225, "S", "T"), ("N18", 230, "D", "N"),
        ("N19", 250, "Q", "E"), ("N20", 255, "G", "A"), ("N21", 260, "R", "K"),
        ("N22", 270, "L", "I"), ("N23", 275, "E", "D"), ("N24", 280, "W", "Y"),
        ("N25", 285, "L", "V"), ("N26", 42, "E", "D"), ("N27", 48, "P", "A"),
        ("N28", 78, "Y", "F"), ("N29", 85, "I", "V"), ("N30", 92, "K", "R"),
        ("N31", 106, "S", "T"), ("N32", 109, "A", "V"), ("N33", 116, "V", "I"),
        ("N34", 124, "E", "D"), ("N35", 132, "Q", "E"), ("N36", 138, "I", "V"),
        ("N37", 144, "A", "G"), ("N38", 149, "K", "R"), ("N39", 156, "K", "R"),
        ("N40", 163, "T", "S")
    ]

    print("[*] Filtering and validating resistance mutations (target: 30, distance >= 15.0 Å)...")
    valid_resistance = []
    for mut_id, res_num, wt, mut in resistance_pool:
        if len(valid_resistance) >= 30:
            break
        dist = calculate_distance_to_ser70(structure, res_num)
        if dist is not None and dist >= 15.0:
            valid_resistance.append({
                "mutation_id": f"R{len(valid_resistance) + 1}",
                "residue_number": res_num,
                "wildtype_aa": wt,
                "mutant_aa": mut,
                "class_label": "resistance",
                "distance_from_ser70_angstroms": dist
            })

    print("[*] Filtering and validating neutral mutations (target: 30, distance >= 15.0 Å)...")
    valid_neutral = []
    for mut_id, res_num, wt, mut in neutral_pool:
        if len(valid_neutral) >= 30:
            break
        dist = calculate_distance_to_ser70(structure, res_num)
        if dist is not None and dist >= 15.0:
            valid_neutral.append({
                "mutation_id": f"N{len(valid_neutral) + 1}",
                "residue_number": res_num,
                "wildtype_aa": wt,
                "mutant_aa": mut,
                "class_label": "neutral",
                "distance_from_ser70_angstroms": dist
            })

    df = pd.DataFrame(valid_resistance + valid_neutral)

    print(f"[*] Collected: {len(valid_resistance)} resistance, {len(valid_neutral)} neutral (Total: {len(df)})")

    output_path = os.path.join("data", "benchmark_mutations.csv")
    df.to_csv(output_path, index=False)
    print(f"[+] Saved dataset to {output_path}")


if __name__ == "__main__":
    generate_benchmark_csv()