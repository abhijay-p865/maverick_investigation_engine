"""Plain-language labels for reports and the website.

Machine-readable identifiers remain stable in JSON and CSV output. These helpers
keep those implementation names out of operator-facing text and charts.
"""

from __future__ import annotations


MEASUREMENT_DISPLAY_NAMES = {
    "elapsed_time_h": "Elapsed batch time",
    "dissolved_oxygen_mg_l": "Dissolved oxygen",
    "ph": "pH",
    "temperature_k": "Temperature",
    "pressure_bar": "Vessel pressure",
    "aeration_lph": "Aeration rate",
    "agitation_rpm": "Agitator speed",
    "substrate_feed_lph": "Substrate feed rate",
    "acid_flow_lph": "Acid addition rate",
    "base_flow_lph": "Base addition rate",
    "coolant_flow_lph": "Cooling-water flow rate",
    "heating_flow_lph": "Heating-water flow rate",
    "water_injection_lph": "Water addition rate",
    "dumped_broth_lph": "Removed broth flow rate",
    "substrate_g_l": "Substrate concentration",
    "penicillin_g_l": "Penicillin concentration",
    "volume_l": "Vessel volume",
    "weight_kg": "Vessel weight",
    "generated_heat_kj": "Generated heat",
    "co2_offgas_pct": "Carbon dioxide in off-gas",
    "paa_flow_lph": "Phenylacetic acid addition rate",
    "paa_offline_g_l": "Offline phenylacetic acid concentration",
    "oil_flow_lph": "Oil addition rate",
    "nh3_offline_g_l": "Offline ammonia concentration",
    "our_g_min": "Oxygen uptake rate",
    "oxygen_offgas_pct": "Oxygen in off-gas",
    "penicillin_offline_g_l": "Offline penicillin concentration",
    "biomass_offline_g_l": "Offline biomass concentration",
    "cer_g_h": "Carbon evolution rate",
    "ammonia_shots_kg": "Ammonia additions",
    "viscosity_offline_cp": "Offline viscosity",
}

STATISTIC_DISPLAY = {
    "mean": (
        "recent average",
        "The average reading during the model's recent analysis window.",
    ),
    "std": (
        "recent variability",
        "How much the readings varied during the model's recent analysis window.",
    ),
    "slope_per_h": (
        "rate of change",
        "How quickly the reading was rising or falling per hour.",
    ),
    "change": (
        "net change",
        "The difference between the first and last reading in the recent analysis window.",
    ),
}


def measurement_display_name(identifier: str) -> str:
    """Return an operator-facing measurement name without units or code syntax."""

    if identifier in MEASUREMENT_DISPLAY_NAMES:
        return MEASUREMENT_DISPLAY_NAMES[identifier]
    return identifier.replace("_", " ").strip().capitalize() or "Process measurement"


def feature_display_name(identifier: str) -> str:
    """Return a readable sensor-pattern name for a model feature identifier."""

    sensor, separator, statistic = identifier.partition("__")
    sensor_name = measurement_display_name(sensor)
    if not separator:
        return sensor_name
    statistic_name = STATISTIC_DISPLAY.get(statistic, ("recent pattern", ""))[0]
    return f"{sensor_name} — {statistic_name}"


def feature_plain_language_meaning(identifier: str) -> str:
    """Explain what a rolling feature measures without exposing its code name."""

    _, separator, statistic = identifier.partition("__")
    if not separator:
        return "A recent measurement pattern used by the model."
    return STATISTIC_DISPLAY.get(
        statistic,
        ("recent pattern", "A recent measurement pattern used by the model."),
    )[1]
