import os

SATELLITE_NAMES = {
    "cloudSAT": "cloudSAT",
    "calipso": "calipso",
    "ICEsat2": "ICEsat2",
    "LRO": "LRO",
    "solarB": "solarB",
}

ORBIT_SUFFIX = "_high.pkl"

MATRICES_DIR = "matrices"


def get_matrix_filename(satellite: str) -> str:
    sat_key = SATELLITE_NAMES.get(satellite)
    if sat_key is None:
        raise ValueError(f"Неизвестный спутник: {satellite}")
    return os.path.join(MATRICES_DIR, f"matrices_{sat_key}{ORBIT_SUFFIX}")
