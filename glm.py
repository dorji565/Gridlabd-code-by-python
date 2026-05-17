import os
import math
import re
import subprocess
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as error:
    np = None
    pd = None
    DATA_DEPENDENCY_ERROR = error
else:
    DATA_DEPENDENCY_ERROR = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


# =========================================================
# 1. PROJECT PATHS
# =========================================================
WORK_DIR = Path(__file__).resolve().parent
GLM_FILE = WORK_DIR / "model.glm"
VOLTAGE_CSV = WORK_DIR / "campus_single_line_voltages.csv"
CURRENT_CSV = WORK_DIR / "campus_single_line_currents.csv"
VOLTAGE_PLOT = WORK_DIR / "voltage_profile.png"
CURRENT_PLOT = WORK_DIR / "current_profile.png"
OPTIMIZATION_PLOT = WORK_DIR / "optimization_rms.png"
OPTIMIZATION_VALIDATION_PLOT = WORK_DIR / "optimization_validation.png"


# =========================================================
# 2. PARAMETERS EDITABLE FROM PYTHON
# =========================================================
params = {
    "scale_load": 1.0,
    "show_plots": True,
    "optimization_steps": 11,
    "source_frequency_hz": 50.0,
    "use_line_inductance": False,
    # UG1: 1.1 kV 4-core 120 sq mm aluminium XLPE cable
    "cfg_ug1_resistance": 0.325000,
    "cfg_ug1_reactance": 0.072000,
    "cfg_ug1_inductance_mH": 0.229183,

    # OH main: ACSR DOG overhead conductor
    "cfg_oh_main_resistance": 0.273000,
    "cfg_oh_main_reactance": 0.350000,
    "cfg_oh_main_inductance_mH": 1.114085,

    "ug1_length": 25.5,
    "oh1_length": 37.6,
    "oh2_length": 130.0,
    "oh3_length": 38.9,
    "oh4_length": 100.0,
    "oh5_length": 56.5,
    "oh6_length": 25.5,
    "oh7_length": 110.0,
    "oh8_length": 34.5,
    "oh9_length": 34.4,
    "oh10_length": 84.9,
    "oh11_length": 88.5,
    "oh12_length": 26.6,
    "oh13_length": 29.9,
    "oh14_length": 27.4,
    "oh15_length": 41.1,
    "oh16_length": 40.8,
    "oh17_length": 23.0,
    "oh18_length": 110.0,
    "oh19_length": 100.0,
    "oh20_length": 110.0,
    "oh21_length": 95.0,
    "oh22_length": 50.0,
    "oh23_length": 105.0,
    "oh24_length": 30.0,
    "oh26_length": 100.0,
    "oh27_length": 35.88,
    "oh28_length": 60.5,
    "oh29_length": 118.0,
    "oh30_length": 36.0,
    "oh31_length": 25.66,
    "oh32_length": 80.1,
    "oh33_length": 90.1,
    "oh34_length": 200.0,
    "oh35_length": 200.0,
}


# Replace these values with measured field voltages from JNEC.
MEASURED_VOLTAGES = {
 # Bus 3
    "bus_3_choekhang": {
        "A": 221.8,
        "B": 220.2,
        "C": 225.506
    },

    # Bus 4
    "bus_4_ece": {
        "A": 220.4,
        "B": 223.6,
        "C": 228.2
    },

    # Bus 6
    "bus_6_jabcdq3": {
        "A": 222.4,
        "B": 223.8,
        "C": 226.92
    },

    # Bus 7
    "bus_7_jefghq2": {
        "A": 222.3,
        "B": 223.8,
        "C": 226.9
    },

    "bus_7_g": {
        "A": 221.2
    },

    # Bus 10
    "bus_10_mb": {
        "A": 222.7,
        "B": 224.8,
        "C": 227.4
    },

    # Bus 11
    "bus_11_aw": {
        "A": 220.1,
        "B": 226.8,
        "C": 225.11
    },

    "bus_11_fmw": {
        "A": 224.4,
        "B": 223.6,
        "C": 226.02
    },

    "bus_11_mph": {
        "A": 225.8,
        "B": 233.1,
        "C": 231.3
    },

    # Bus 13
    "bus_13_e5": {
        "A": 224.7
    },

    # Bus 14
    "bus_14_e4": {
        "A": 221.7
    },

    # Bus 15
    "bus_15_e3": {
        "A": 225.8
    },

    "bus_15_c": {
        "A": 223.9
    },

    # Bus 16
    "bus_16_e1": {
        "A": 221.6
    },

    "bus_16_e2": {
        "A": 222.6
    },

    # Bus 19
    "bus_19_c5": {
        "A": 221.7,
        "B": 221.9,
        "C": 222.346
    },

    "bus_19_c6": {
        "A": 222.1
    },

    # Bus 20
    "bus_20_ad": {
        "A": 221.3,
        "B": 226.5,
        "C": 224.3
    },

    "bus_20_ed": {
        "A": 220.4,
        "B": 223.6,
        "C": 228.2
    },

    "bus_20_cs": {
        "A": 226.2
    },

    # Bus 22
    "bus_22_mess": {
        "A": 220.0,
        "B": 222.0,
        "C": 223.83
    },

    # Bus 23
    "bus_23_na": {
        "A": 219.5,
        "B": 228.6,
        "C": 183.8
    },

    # Bus 24
    "bus_24_nb": {
        "A": 220.2,
        "B": 228.2,
        "C": 219.7
    },

    # Bus 25
    "bus_25_cb": {
        "A": 222.2,
        "B": 222.6,
        "C": 225.3
    },

    "bus_25_cf": {
        "A": 223.9
    },

    "bus_25_pq": {
        "A": 225.7
    },

    # Bus 26
    "bus_26_cw": {
        "A": 222.3,
        "B": 221.8,
        "C": 222.018
    },

    "bus_26_smwl": {
        "A": 220.7,
        "B": 221.0,
        "C": 222.59
    },

    # Bus 28
    "bus_28_c1": {
        "A": 225.4
    },

    # Bus 29
    "bus_29_b1": {
        "A": 221.8,
        "B": 220.5
    },

    "bus_29_b2": {
        "A": 221.1,
        "B": 222.1
    },

    # Bus 30
    "bus_30_c2": {
        "A": 227.2
    },

    # Bus 31
    "bus_31_c3": {
        "A": 223.8
    },

    "bus_31_c4": {
        "A": 221.3
    },

    # Bus 32
    "bus_32_dit_library": {
        "A": 218.0,
        "B": 220.2,
        "C": 224.5
    },

    # Bus 33
    "bus_33_r1": {
        "A": 231.8,
        "B": 232.0
    },

    # Bus 34
    "bus_34_rb2": {
        "A": 224.6
    },
    # Bus 35
    "bus_35_PQ": {
        "A": 226
    },

    # Bus 36
    "bus_36_gh": {
        "A": 223.8
    }
}


# =========================================================
# 3. NETWORK DATA - JNEC DISTRIBUTION MODEL
# =========================================================
BUSES = [
    "bus_2", "bus_3", "bus_4", "bus_5", "bus_6", "bus_7", "bus_8",
    "bus_9", "bus_10", "bus_11", "bus_12", "bus_13", "bus_14",
    "bus_15", "bus_16", "bus_17", "bus_18", "bus_19", "bus_20",
    "bus_21", "bus_22", "bus_23", "bus_24", "bus_25", "bus_26", 
    "bus_27", "bus_28", "bus_29", "bus_30", "bus_31", "bus_32", 
    "bus_33", "bus_34", "bus_35", "bus_36",
]

LINES = [
    ("ug1", "underground_line", "ABCN", "bus_1_lv", "bus_2", "ug1_length", "cfg_ug1"),
    ("oh1", "overhead_line", "ABCN", "bus_2", "bus_3", "oh1_length", "cfg_oh_main"),
    ("oh2", "overhead_line", "ABCN", "bus_3", "bus_4", "oh2_length", "cfg_oh_main"),
    ("oh3", "overhead_line", "ABCN", "bus_4", "bus_5", "oh3_length", "cfg_oh_main"),
    ("oh4", "overhead_line", "ABCN", "bus_5", "bus_6", "oh4_length", "cfg_oh_main"),
    ("oh5", "overhead_line", "ABCN", "bus_6", "bus_7", "oh5_length", "cfg_oh_main"),
    ("oh6", "overhead_line", "ABCN", "bus_5", "bus_8", "oh6_length", "cfg_oh_main"),
    ("oh7", "overhead_line", "ABCN", "bus_2", "bus_9", "oh7_length", "cfg_oh_main"),
    ("oh8", "overhead_line", "ABCN", "bus_9", "bus_10", "oh8_length", "cfg_oh_main"),
    ("oh9", "overhead_line", "ABCN", "bus_10", "bus_11", "oh9_length", "cfg_oh_main"),
    ("oh10", "overhead_line", "ABCN", "bus_11", "bus_12", "oh10_length", "cfg_oh_main"),
    ("oh11", "overhead_line", "ABCN", "bus_12", "bus_13", "oh11_length", "cfg_oh_main"),
    ("oh12", "overhead_line", "ABCN", "bus_13", "bus_14", "oh12_length", "cfg_oh_main"),
    ("oh13", "overhead_line", "ABCN", "bus_14", "bus_15", "oh13_length", "cfg_oh_main"),
    ("oh14", "overhead_line", "ABCN", "bus_15", "bus_16", "oh14_length", "cfg_oh_main"),
    ("oh15", "overhead_line", "ABCN", "bus_16", "bus_17", "oh15_length", "cfg_oh_main"),
    ("oh16", "overhead_line", "ABCN", "bus_17", "bus_18", "oh16_length", "cfg_oh_main"),
    ("oh17", "overhead_line", "ABCN", "bus_18", "bus_19", "oh17_length", "cfg_oh_main"),
    ("oh18", "overhead_line", "ABCN", "bus_12", "bus_20", "oh18_length", "cfg_oh_main"),
    ("oh19", "overhead_line", "ABCN", "bus_2", "bus_21", "oh19_length", "cfg_oh_main"),
    ("oh20", "overhead_line", "ABCN", "bus_21", "bus_22", "oh20_length", "cfg_oh_main"),
    ("oh21", "overhead_line", "ABCN", "bus_22", "bus_23", "oh21_length", "cfg_oh_main"),
    ("oh22", "overhead_line", "ABCN", "bus_23", "bus_24", "oh22_length", "cfg_oh_main"),
    ("oh23", "overhead_line", "ABCN", "bus_21", "bus_25", "oh23_length", "cfg_oh_main"),
    ("oh24", "overhead_line", "ABCN", "bus_25", "bus_26", "oh24_length", "cfg_oh_main"),
    ("oh26", "overhead_line", "ABCN", "bus_26", "bus_27", "oh26_length", "cfg_oh_main"),
    ("oh27", "overhead_line", "ABCN", "bus_27", "bus_28", "oh27_length", "cfg_oh_main"),
    ("oh28", "overhead_line", "ABCN", "bus_28", "bus_29", "oh28_length", "cfg_oh_main"),
    ("oh29", "overhead_line", "ABCN", "bus_29", "bus_30", "oh29_length", "cfg_oh_main"),
    ("oh30", "overhead_line", "ABCN", "bus_30", "bus_31", "oh30_length", "cfg_oh_main"),
    ("oh31", "overhead_line", "ABCN", "bus_31", "bus_32", "oh31_length", "cfg_oh_main"),
    ("oh32", "overhead_line", "ABCN", "bus_27", "bus_33", "oh32_length", "cfg_oh_main"),
    ("oh33", "overhead_line", "ABCN", "bus_27", "bus_34", "oh33_length", "cfg_oh_main"),
    ("oh34", "overhead_line", "ABCN", "bus_34", "bus_35", "oh34_length", "cfg_oh_main"),
    ("oh35", "overhead_line", "ABCN", "bus_35", "bus_36", "oh35_length", "cfg_oh_main"),
]

LOADS = [
    ("load_bus3_lt", "bus_3", "ABCN", {"A": (20.0, 13.0), "B": (10.0, 12.0), "C": (10.0, 14.0)}),
    ("load_bus3_choekhang", "bus_3", "ABCN", {"A": (40.0, 26.0), "B": (20.0, 24.0), "C": (216.0, 28.0)}),
    ("load_bus4_ece", "bus_4", "ABCN", {"A": (184.0, 134.0), "B": (238.0, 202.0), "C": (572.0, 376.0)}),
    ("load_bus6_ja_jd", "bus_6", "ABCN", {"A": (86.0, 36.0), "B": (158.0, 10.0), "C": (620.0, 56.0)}),
    ("load_bus7_je_jg", "bus_7", "ABCN", {"A": (208.0, 32.0), "B": (154.0, 13.8), "C": (600.0, 18.0)}),
    ("load_bus8_ed", "bus_8", "ABCN", {"A": (184.0, 134.0), "B": (238.0, 202.0), "C": (572.0, 376.0)}),
    ("load_bus9_zb", "bus_9", "ABCN", {"A": (86.0, 36.0), "B": (40.0, 10.0), "C": (100.0, 56.0)}),
    ("load_bus10_mb", "bus_10", "ABCN", {"A": (1452.0, 330.0), "B": (676.0, 250.0), "C": (1528.0, 268.0)}),
    ("load_bus11_aw_fmw_mph", "bus_11", "ABCN", {"A": (206.0, 564.0), "B": (34.0, 4.0), "C": (300.0, 84.0)}),
    ("load_bus13_e5", "bus_13", "AN", {"A": (314.0, 88.0)}),
    ("load_bus14_e4", "bus_14", "AN", {"A": (346.0, 98.2)}),
    ("load_bus15_e3_civilb", "bus_15", "AN", {"A": (374.0, 66.2)}),
    ("load_bus16_e1_e2", "bus_16", "AN", {"A": (368.0, 77.0)}),
    ("load_bus18_a1_a2_a3", "bus_18", "ABCN", {"A": (3012.667, 2259.5), "B": (3012.667, 2259.5), "C": (3012.667, 2259.5)}),
    ("load_bus19_c5_c6", "bus_19", "ABCN", {"A": (60.2, 28.8), "B": (60.0, 15.2), "C": (54.0, 22.0)}),
    ("load_bus20_ad_css", "bus_20", "ABCN", {"A": (640.0, 124.0), "B": (440.0, 254.0), "C": (722.0, 80.0)}),
    ("load_bus22_upper_missi", "bus_22", "ABCN", {"A": (5330.0, 78.0), "B": (4660.0, 94.0), "C": (5200.0, 226.0)}),
    ("load_bus23_na", "bus_23", "ABCN", {"A": (2030.0, 106.2), "B": (280.0, 76.0), "C": (268.0, 42.0)}),
    ("load_bus24_nb", "bus_24", "ABCN", {"A": (1220.0, 156.0), "B": (970.0, 118.0), "C": (1766.0, 146.0)}),
    ("load_bus25_canteen", "bus_25", "ABCN", {"A": (1824.0, 874.0), "B": (130.0, 44.0), "C": (0.0, 0.0)}),
    ("load_bus26_smwl_cw", "bus_26", "ABCN", {"A": (1210.0, 106.6), "B": (40.0, 26.4), "C": (26.0, 67.0)}),
    ("load_bus28_c1", "bus_28", "AN", {"A": (170.0, 180.0)}),
    ("load_bus29_b1_b2", "bus_29", "ABN", {"A": (1980.0, 148.8), "B": (230.0, 80.8)}),
    ("load_bus30_cate_c2", "bus_30", "AN", {"A": (180.0, 26.0)}),
    ("load_bus31_c3_c4", "bus_31", "AN", {"A": (370.0, 86.0)}),
    ("load_bus32_bit_library", "bus_32", "ABCN", {"A": (3400.0, 748.0), "B": (2460.0, 640.0), "C": (2392.0, 262.0)}),
    ("load_bus33_r81", "bus_33", "AN", {"A": (180.0, 132.0)}),
    ("load_bus34_r82_a1", "bus_34", "AN", {"A": (180.0, 132.0)}),
    ("load_bus35_PQ", "bus_35", "AN", {"A": (290, 134)}),
    ("load_bus36_gh_gate", "bus_36", "AN", {"A": (9520.0, 7140.0)}),
]


# =========================================================
# 4. GLM GENERATION
# =========================================================
def complex_power_text(real_power, reactive_power, scale):
    return f"{real_power * scale:.6f}+{reactive_power * scale:.6f}j"


def line_reactance(config_name, active_params):
    if active_params["use_line_inductance"]:
        frequency = active_params["source_frequency_hz"]
        inductance_h = active_params[f"{config_name}_inductance_mH"] / 1000.0
        return 2.0 * math.pi * frequency * inductance_h

    return active_params[f"{config_name}_reactance"]


def line_impedance_text(config_name, active_params):
    resistance = active_params[f"{config_name}_resistance"]
    reactance = line_reactance(config_name, active_params)
    return f"{resistance:.6f}+{reactance:.6f}j"


def render_line(line, active_params):
    name, object_type, phases, from_bus, to_bus, length_key, config = line
    length = active_params.get(length_key, 0)
    return f"""object {object_type} {{
    name {name};
    phases "{phases}";
    from {from_bus};
    to {to_bus};
    length {length};
    configuration {config};
}}"""


def render_load(load, active_params):
    name, parent, phases, powers = load
    scale = active_params["scale_load"]
    power_lines = []

    for phase in ("A", "B", "C"):
        if phase in powers:
            p, q = powers[phase]
            power_lines.append(f"    constant_power_{phase} {complex_power_text(p, q, scale)};")

    power_text = "\n".join(power_lines)
    return f"""object load {{
    name {name};
    parent {parent};
    phases "{phases}";
    nominal_voltage 230.940108;
{power_text}
}}"""


def generate_glm(filename=GLM_FILE, active_params=None):
    active_params = params if active_params is None else active_params

    transformer_resistance = 0.01
    transformer_reactance = 0.04899
    ug1_impedance = line_impedance_text("cfg_ug1", active_params)
    oh_main_impedance = line_impedance_text("cfg_oh_main", active_params)

    bus_text = "\n".join(
        f'object node {{ name {bus}; phases "ABCN"; nominal_voltage 230.940108; }}'
        for bus in BUSES
    )
    line_text = "\n\n".join(render_line(line, active_params) for line in LINES)
    load_text = "\n\n".join(render_load(load, active_params) for load in LOADS)

    glm = f"""
clock {{
    timezone UTC0;
    starttime '2026-01-01 00:00:00';
    stoptime '2026-01-01 00:00:01';
}}

module tape;

module powerflow {{
    solver_method FBS;
    line_capacitance false;
}}

object voltdump {{
    filename "{VOLTAGE_CSV.name}";
    mode POLAR;
}}

object currdump {{
    filename "{CURRENT_CSV.name}";
    mode POLAR;
}}

// 11 kV source bus shown as Bus 1
object node {{
    name bus_1_primary;
    phases "ABC";
    bustype SWING;
    nominal_voltage 6350.852961;
    voltage_A 6350.852961+0j;
    voltage_B -3175.426481-5500.000000j;
    voltage_C -3175.426481+5500.000000j;
}}

// Low-voltage transformer secondary before UG1
object node {{
    name bus_1_lv;
    phases "ABCN";
    nominal_voltage 230.940108;
    voltage_A 230.940108+0j;
    voltage_B -115.470054-200.000000j;
    voltage_C -115.470054+200.000000j;
}}

object transformer_configuration {{
    name cfg_tf_bus1;
    connect_type DELTA_GWYE;
    install_type PADMOUNT;
    power_rating 500.0;
    primary_voltage 11000.0;
    secondary_voltage 400.0;
    resistance {transformer_resistance};
    reactance {transformer_reactance};
}}

object transformer {{
    name tf_bus1;
    phases "ABCN";
    from bus_1_primary;
    to bus_1_lv;
    configuration cfg_tf_bus1;
}}

// Reusable simplified line impedance models
object line_configuration {{
    name cfg_ug1;
    z11 {ug1_impedance};
    z12 0+0j;
    z13 0+0j;
    z21 0+0j;
    z22 {ug1_impedance};
    z23 0+0j;
    z31 0+0j;
    z32 0+0j;
    z33 {ug1_impedance};
}}

object line_configuration {{
    name cfg_oh_main;
    z11 {oh_main_impedance};
    z12 0+0j;
    z13 0+0j;
    z21 0+0j;
    z22 {oh_main_impedance};
    z23 0+0j;
    z31 0+0j;
    z32 0+0j;
    z33 {oh_main_impedance};
}}

// Bus objects from the single-line diagram
{bus_text}

// Distribution lines
{line_text}

// Loads
{load_text}
"""

    Path(filename).write_text(glm.strip() + "\n", encoding="utf-8")
    return Path(filename)


# =========================================================
# 5. RUN GRIDLAB-D
# =========================================================
def run_simulation(glm_file=GLM_FILE):
    try:
        result = subprocess.run(
            ["gridlabd", str(glm_file)],
            cwd=WORK_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        return True
    except FileNotFoundError:
        print("GridLAB-D was not found. Add gridlabd.exe to PATH and try again.")
        return False
    except subprocess.CalledProcessError as error:
        print("Simulation failed.")
        if error.stdout:
            print(error.stdout)
        if error.stderr:
            print(error.stderr)
        return False


def require_data_dependencies():
    if DATA_DEPENDENCY_ERROR is not None:
        missing = DATA_DEPENDENCY_ERROR.name
        raise ModuleNotFoundError(
            f"Missing required package '{missing}'. Install required packages with: "
            "python -m pip install numpy pandas matplotlib"
        )


# =========================================================
# 6. READ OUTPUT CSV
# =========================================================
def read_results():
    require_data_dependencies()

    if not VOLTAGE_CSV.exists():
        raise FileNotFoundError(f"Voltage CSV not found: {VOLTAGE_CSV}")
    if not CURRENT_CSV.exists():
        raise FileNotFoundError(f"Current CSV not found: {CURRENT_CSV}")

    volt = pd.read_csv(VOLTAGE_CSV, comment="#")
    curr = pd.read_csv(CURRENT_CSV, comment="#")
    volt.columns = volt.columns.str.strip()
    curr.columns = curr.columns.str.strip()
    return volt, curr


# =========================================================
# 7. RMS VALIDATION
# =========================================================
def rms_error(simulated, measured):
    require_data_dependencies()

    simulated = np.asarray(simulated, dtype=float)
    measured = np.asarray(measured, dtype=float)
    return float(np.sqrt(np.mean((simulated - measured) ** 2)))


def resolve_voltage_node_name(measured_name, available_node_names):
    if measured_name in available_node_names:
        return measured_name

    match = re.match(r"^(bus_\d+)", str(measured_name))
    if match:
        bus_name = match.group(1)
        if bus_name in available_node_names:
            return bus_name
        if bus_name == "bus_1" and "bus_1_lv" in available_node_names:
            return "bus_1_lv"

    load_name = f"load_{measured_name}"
    if load_name in available_node_names:
        return load_name

    return None


def validation_vectors(volt, measured_voltages):
    require_data_dependencies()

    simulated = []
    measured = []
    records = []
    available_node_names = set(volt["node_name"].astype(str))
    missing_names = []

    for node_name, phase_values in measured_voltages.items():
        resolved_node_name = resolve_voltage_node_name(node_name, available_node_names)
        if resolved_node_name is None:
            missing_names.append(node_name)
            continue

        row = volt.loc[volt["node_name"].astype(str) == resolved_node_name]
        if row.empty:
            missing_names.append(node_name)
            continue

        row = row.iloc[0]
        for phase, measured_value in phase_values.items():
            col = f"volt{phase}_mag"
            if col not in volt.columns:
                print(f"Column {col} is not present in voltage CSV. Skipping.")
                continue

            simulated_value = float(row[col])
            simulated.append(simulated_value)
            measured.append(float(measured_value))
            records.append(
                {
                    "measured_name": node_name,
                    "simulated_node": resolved_node_name,
                    "phase": phase,
                    "simulated_voltage": simulated_value,
                    "measured_voltage": float(measured_value),
                    "error": simulated_value - float(measured_value),
                }
            )

    if missing_names:
        print(
            f"Skipped {len(missing_names)} measured points because no matching bus "
            "was found in the voltage CSV."
        )

    return np.array(simulated), np.array(measured), pd.DataFrame(records)


def validate_voltage(volt, measured_voltages=MEASURED_VOLTAGES):
    simulated, measured, table = validation_vectors(volt, measured_voltages)

    if len(simulated) == 0:
        print("No validation points found. Add measured voltages to MEASURED_VOLTAGES.")
        return None, table

    error = rms_error(simulated, measured)
    print("\nRMS voltage validation")
    print(table.to_string(index=False))
    print(f"RMS Error: {error:.6f} V")
    return error, table


# =========================================================
# 8. ENHANCED PLOTTING WITH OPTIMIZATION VALIDATION
# =========================================================
def plot_3phase(volt, curr=None, show=True):

    print("\nGenerating 3-phase plots...")

    if plt is None:
        print("matplotlib is not installed.")
        return

    # =====================================================
    # VOLTAGE PROFILE
    # =====================================================

    voltage_cols = ["voltA_mag", "voltB_mag", "voltC_mag"]

    missing_voltage_cols = [
        col for col in voltage_cols if col not in volt.columns
    ]

    if missing_voltage_cols:
        print("Voltage columns missing:", missing_voltage_cols)
        return

    if "node_name" not in volt.columns:
        print("node_name column missing.")
        return

    voltage_plot_data = volt[
        volt["node_name"].astype(str).str.startswith("bus_")
        & (volt["node_name"] != "bus_1_primary")
    ].copy()

    voltage_plot_data["bus_label"] = voltage_plot_data[
        "node_name"
    ].replace(
        {"bus_1_lv": "bus_1"}
    )

    def bus_sort_key(bus_name):
        return int(str(bus_name).split("_")[1])

    voltage_plot_data["bus_order"] = voltage_plot_data[
        "bus_label"
    ].apply(bus_sort_key)

    voltage_plot_data = voltage_plot_data.sort_values("bus_order")

    x_voltage = voltage_plot_data["bus_label"].tolist()

    phase_a = voltage_plot_data["voltA_mag"].to_numpy()
    phase_b = voltage_plot_data["voltB_mag"].to_numpy()
    phase_c = voltage_plot_data["voltC_mag"].to_numpy()

    x_positions = np.arange(len(x_voltage))
    bar_width = 0.16

    fig, ax = plt.subplots(figsize=(22, 8))

    ax.bar(
        x_positions - bar_width,
        phase_a,
        width=bar_width,
        color="red",
        label="R phase"
    )

    ax.bar(
        x_positions,
        phase_b,
        width=bar_width,
        color="yellow",
        edgecolor="gold",
        linewidth=0.4,
        label="Y phase"
    )

    ax.bar(
        x_positions + bar_width,
        phase_c,
        width=bar_width,
        color="#5b9bd5",
        label="B phase"
    )

    ax.set_title(
        "Voltage profile",
        fontsize=26,
        color="#555555",
        pad=28
    )

    ax.set_xlabel(
        "BUSES (node name)",
        fontsize=14,
        color="#555555",
        labelpad=10
    )

    ax.set_ylabel(
        "VOLTAGE VALUES OF EACH PHASE (V)",
        fontsize=14,
        color="#555555"
    )

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        x_voltage,
        rotation=90,
        fontsize=8
    )

    ax.set_ylim(218, 232)

    ax.set_yticks(np.arange(218, 233, 2))

    ax.tick_params(
        axis="y",
        labelsize=9,
        colors="#555555"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=8
    )

    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)

    ax.grid(axis="x", visible=False)

    ax.set_axisbelow(True)

    ax.spines["left"].set_color("#d9d9d9")
    ax.spines["bottom"].set_color("#d9d9d9")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    plt.savefig(VOLTAGE_PLOT, dpi=200)
    # =====================================================
    # CURRENT PROFILE
    # =====================================================

    if curr is not None:

        current_cols = [
            "currA_mag",
            "currB_mag",
            "currC_mag"
        ]

        missing_current_cols = [
            col for col in current_cols
            if col not in curr.columns
        ]

        if missing_current_cols:

            print(
                "Current columns missing:",
                missing_current_cols
            )

        else:

            if "link_name" in curr.columns:
                x_current = curr["link_name"].astype(str).tolist()
            else:
                x_current = [f"L{i+1}" for i in range(len(curr))]

            phase_a_current = curr["currA_mag"].to_numpy()
            phase_b_current = curr["currB_mag"].to_numpy()
            phase_c_current = curr["currC_mag"].to_numpy()

            x_positions_current = np.arange(len(x_current))

            bar_width_current = 0.16

            fig2, ax2 = plt.subplots(figsize=(22, 8))

            ax2.bar(
                x_positions_current - bar_width_current,
                phase_a_current,
                width=bar_width_current,
                color="red",
                label="R phase"
            )

            ax2.bar(
                x_positions_current,
                phase_b_current,
                width=bar_width_current,
                color="yellow",
                edgecolor="gold",
                linewidth=0.4,
                label="Y phase"
            )

            ax2.bar(
                x_positions_current + bar_width_current,
                phase_c_current,
                width=bar_width_current,
                color="#5b9bd5",
                label="B phase"
            )

            ax2.set_title(
                "Current profile",
                fontsize=26,
                color="#555555",
                pad=28
            )

            ax2.set_xlabel(
                "LINES (line name)",
                fontsize=14,
                color="#555555",
                labelpad=10
            )

            ax2.set_ylabel(
                "CURRENT VALUES OF EACH PHASE (A)",
                fontsize=14,
                color="#555555"
            )

            ax2.set_xticks(x_positions_current)

            ax2.set_xticklabels(
                x_current,
                rotation=90,
                fontsize=8
            )

            ax2.tick_params(
                axis="y",
                labelsize=9,
                colors="#555555"
            )

            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.05),
                ncol=3,
                frameon=False,
                fontsize=8
            )

            ax2.grid(
                axis="y",
                color="#d9d9d9",
                linewidth=0.8
            )

            ax2.grid(axis="x", visible=False)

            ax2.set_axisbelow(True)

            ax2.spines["left"].set_color("#d9d9d9")
            ax2.spines["bottom"].set_color("#d9d9d9")

            for spine in ("top", "right"):
                ax2.spines[spine].set_visible(False)


            plt.tight_layout()

            plt.savefig(CURRENT_PLOT, dpi=200)

    if show:
        plt.show()
    else:
        plt.close("all")

    print("Plotting complete.")


def plot_optimization(history, show=True):
    """Plot RMS error vs load scale"""
    if not history:
        print("No optimization history available to plot.")
        return

    if plt is None:
        print("matplotlib is not installed. Skipping optimization plot.")
        return

    scales = [item["scale_load"] for item in history]
    errors = [item["rms_error"] for item in history]

    # Find minimum error (best optimization point)
    min_idx = np.argmin(errors)
    best_scale = scales[min_idx]
    best_error = errors[min_idx]

    plt.figure(figsize=(12, 7))
    
    # Plot the curve
    plt.plot(
        scales,
        errors,
        marker="o",
        color="tab:blue",
        linewidth=2.5,
        markersize=8,
        label="RMS error",
    )
    
    # Highlight the optimization point
    plt.plot(
        best_scale,
        best_error,
        marker="*",
        color="red",
        markersize=30,
        label=f"Optimal Point (Scale={best_scale:.3f}, RMS={best_error:.4f}V)",
        zorder=5
    )
    
    # Add annotation for optimal point
    plt.annotate(
        f"Optimal\nScale: {best_scale:.3f}\nRMS: {best_error:.4f} V",
        xy=(best_scale, best_error),
        xytext=(best_scale + 0.1, best_error + 0.3),
        fontsize=11,
        fontweight='bold',
        color='red',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', color='red', lw=2)
    )
    
    plt.title(
        "Load Scale Optimization - RMS Voltage Error",
        fontsize=16,
        color="#555555",
        fontweight='bold'
    )
    plt.xlabel("Load Scale Factor (p.u.)", fontsize=13, fontweight='bold', color="#555555")
    plt.ylabel("RMS Voltage Error (V)", fontsize=13, fontweight='bold', color="#555555")
    plt.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    plt.grid(axis="x", color="#d9d9d9", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.savefig(OPTIMIZATION_PLOT, dpi=200)
    print(f"Optimization plot saved to: {OPTIMIZATION_PLOT}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_optimization_validation(best_scale, volt, measured_voltages=MEASURED_VOLTAGES, show=True):
    """Plot simulated vs measured voltages at optimal scale with intersection line"""
    if plt is None:
        print("matplotlib is not installed. Skipping validation plot.")
        return

    require_data_dependencies()
    
    simulated, measured, table = validation_vectors(volt, measured_voltages)
    
    if len(simulated) == 0:
        print("No validation data to plot.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ======== LEFT PLOT: Simulated vs Measured Voltages ========
    x_indices = np.arange(len(simulated))
    
    ax1.plot(
        x_indices,
        simulated,
        marker="o",
        color="blue",
        linewidth=2.5,
        markersize=7,
        label="Simulated Voltage",
        alpha=0.8
    )
    
    ax1.plot(
        x_indices,
        measured,
        marker="s",
        color="green",
        linewidth=2.5,
        markersize=7,
        label="Measured Voltage",
        alpha=0.8
    )
    
    # Find intersection or closest points
    differences = np.abs(simulated - measured)
    closest_idx = np.argmin(differences)
    
    # Plot intersection point
    ax1.plot(
        closest_idx,
        simulated[closest_idx],
        marker="*",
        color="red",
        markersize=25,
        label="Intersection Point",
        zorder=5
    )
    
    ax1.annotate(
        f"Optimal Match Point\nIndex: {closest_idx}\nSim: {simulated[closest_idx]:.2f}V\nMeas: {measured[closest_idx]:.2f}V\nError: {differences[closest_idx]:.4f}V",
        xy=(closest_idx, simulated[closest_idx]),
        xytext=(closest_idx + 3, simulated[closest_idx] + 1),
        fontsize=10,
        fontweight='bold',
        color='red',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', color='red', lw=2)
    )
    
    ax1.set_xlabel("Measurement Point Index (index)", fontsize=12, fontweight='bold', color="#555555")
    ax1.set_ylabel("Voltage (V)", fontsize=12, fontweight='bold', color="#555555")
    ax1.set_title(
        f"Simulated vs Measured Voltages\n(Load Scale = {best_scale:.3f})",
        fontsize=13,
        fontweight='bold',
        color="#555555"
    )
    ax1.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax1.legend(fontsize=11, loc='best')

    # ======== RIGHT PLOT: Error Distribution ========
    errors = simulated - measured
    
    ax2.bar(
        x_indices,
        errors,
        color=['red' if e > 0 else 'green' for e in errors],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add zero reference line
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error')
    
    ax2.set_xlabel("Measurement Point Index (index)", fontsize=12, fontweight='bold', color="#555555")
    ax2.set_ylabel("Error, Simulated - Measured (V)", fontsize=12, fontweight='bold', color="#555555")
    ax2.set_title(
        "Voltage Error Distribution\n(Positive = Over-prediction, Negative = Under-prediction)",
        fontsize=13,
        fontweight='bold',
        color="#555555"
    )
    ax2.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7, axis='y')
    ax2.legend(fontsize=11, loc='best')
    
    # Add RMS error text
    rms_err = rms_error(simulated, measured)
    fig.text(0.5, 0.02, f'Overall RMS Error: {rms_err:.4f} V', 
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OPTIMIZATION_VALIDATION_PLOT, dpi=200)
    print(f"Optimization validation plot saved to: {OPTIMIZATION_VALIDATION_PLOT}")

    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# 9. OPTIMIZATION LOOP
# =========================================================
def optimize_load_scale(scale_min=0.5, scale_max=1.5, steps=11, show_plots=True):
    print("\nStarting optimization...")
    require_data_dependencies()

    original_scale = params["scale_load"]
    best_error = float("inf")
    best_scale = original_scale
    best_volt = None
    history = []

    for scale in np.linspace(scale_min, scale_max, steps):
        params["scale_load"] = float(scale)
        generate_glm(GLM_FILE, params)

        if not run_simulation(GLM_FILE):
            continue

        volt, _ = read_results()
        error, _ = validate_voltage(volt, MEASURED_VOLTAGES)

        if error is None:
            print("Optimization stopped because no measured validation points are available.")
            params["scale_load"] = original_scale
            return best_scale, best_error, history, best_volt

        history.append({"scale_load": float(scale), "rms_error": error})
        print(f"Scale: {scale:.3f}, RMS Error: {error:.6f} V")

        if error < best_error:
            best_error = error
            best_scale = float(scale)
            best_volt = volt

    params["scale_load"] = best_scale
    generate_glm(GLM_FILE, params)

    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Optimal load scale: {best_scale:.4f}")
    print(f"Minimum RMS error: {best_error:.6f} V")
    print("="*70)
    
    plot_optimization(history, show=show_plots)
    
    # Plot validation comparison at optimal scale
    if best_volt is not None:
        plot_optimization_validation(best_scale, best_volt, MEASURED_VOLTAGES, show=show_plots)
    
    return best_scale, best_error, history, best_volt


# =========================================================
# 10. MAIN EXECUTION
# =========================================================
def main():
    os.chdir(WORK_DIR)
    print("Starting JNEC GridLAB-D simulation workflow...")
    print(f"Working directory: {WORK_DIR}")

    generate_glm(GLM_FILE, params)
    print(f"Generated GLM file: {GLM_FILE}")

    if not run_simulation(GLM_FILE):
        print("Simulation failed. Fix the generated GLM before continuing.")
        return

    if DATA_DEPENDENCY_ERROR is not None:
        print("Simulation completed, but Python data-analysis packages are missing.")
        print(f"Missing package: {DATA_DEPENDENCY_ERROR.name}")
        print("Install required packages with: python -m pip install numpy pandas matplotlib")
        return

    volt, curr = read_results()

    print("\nCSV loaded successfully.")
    print("\nVoltage CSV preview:")
    print(volt.head().to_string(index=False))

    validate_voltage(volt, MEASURED_VOLTAGES)
    plot_3phase(volt, curr, show=bool(params["show_plots"]))
    
    optimize_load_scale(
        scale_min=0.5,
        scale_max=1.5,
        steps=int(params["optimization_steps"]),
        show_plots=bool(params["show_plots"]),
    )

    print("\nWorkflow complete.")
    print(f"Final optimized GLM file: {GLM_FILE}")


if __name__ == "__main__":
    main()
