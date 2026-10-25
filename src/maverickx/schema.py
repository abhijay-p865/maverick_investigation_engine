"""Documented names and units used inside MaverickX."""

from __future__ import annotations

REQUIRED_MEASUREMENTS = (
    "dissolved_oxygen_mg_l",
    "ph",
    "temperature_k",
    "pressure_bar",
)

REQUIRED_COLUMNS = ("batch_id", "elapsed_time_h", *REQUIRED_MEASUREMENTS)

INDPENSIM_COLUMN_MAP = {
    "Time (h)": "elapsed_time_h",
    "Aeration rate(Fg:L/h)": "aeration_lph",
    "Agitator RPM(RPM:RPM)": "agitation_rpm",
    "Sugar feed rate(Fs:L/h)": "substrate_feed_lph",
    "Acid flow rate(Fa:L/h)": "acid_flow_lph",
    "Base flow rate(Fb:L/h)": "base_flow_lph",
    "Heating/cooling water flow rate(Fc:L/h)": "coolant_flow_lph",
    "Heating water flow rate(Fh:L/h)": "heating_flow_lph",
    "Water for injection/dilution(Fw:L/h)": "water_injection_lph",
    "Air head pressure(pressure:bar)": "pressure_bar",
    "Dumped broth flow(Fremoved:L/h)": "dumped_broth_lph",
    "Substrate concentration(S:g/L)": "substrate_g_l",
    "Dissolved oxygen concentration(DO2:mg/L)": "dissolved_oxygen_mg_l",
    "Penicillin concentration(P:g/L)": "penicillin_g_l",
    "Vessel Volume(V:L)": "volume_l",
    "Vessel Weight(Wt:Kg)": "weight_kg",
    "pH(pH:pH)": "ph",
    "Temperature(T:K)": "temperature_k",
    "Generated heat(Q:kJ)": "generated_heat_kj",
    "carbon dioxide percent in off-gas(CO2outgas:%)": "co2_offgas_pct",
    "PAA flow(Fpaa:PAA flow (L/h))": "paa_flow_lph",
    "PAA concentration offline(PAA_offline:PAA (g L^{-1}))": "paa_offline_g_l",
    "Oil flow(Foil:L/hr)": "oil_flow_lph",
    "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))": "nh3_offline_g_l",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))": "our_g_min",
    "Oxygen in percent in off-gas(O2:O2  (%))": "oxygen_offgas_pct",
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))": "penicillin_offline_g_l",
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))": "biomass_offline_g_l",
    "Carbon evolution rate(CER:g/h)": "cer_g_h",
    "Ammonia shots(NH3_shots:kgs)": "ammonia_shots_kg",
    "Viscosity(Viscosity_offline:centPoise)": "viscosity_offline_cp",
}

UNITS = {
    "elapsed_time_h": "h",
    "aeration_lph": "L/h",
    "agitation_rpm": "rpm",
    "substrate_feed_lph": "L/h",
    "acid_flow_lph": "L/h",
    "base_flow_lph": "L/h",
    "coolant_flow_lph": "L/h",
    "heating_flow_lph": "L/h",
    "water_injection_lph": "L/h",
    "pressure_bar": "bar",
    "dumped_broth_lph": "L/h",
    "substrate_g_l": "g/L",
    "dissolved_oxygen_mg_l": "mg/L",
    "penicillin_g_l": "g/L",
    "volume_l": "L",
    "weight_kg": "kg",
    "ph": "pH",
    "temperature_k": "K",
    "generated_heat_kj": "kJ",
    "co2_offgas_pct": "%",
    "paa_flow_lph": "L/h",
    "paa_offline_g_l": "g/L",
    "oil_flow_lph": "L/h",
    "nh3_offline_g_l": "g/L",
    "our_g_min": "g/min",
    "oxygen_offgas_pct": "%",
    "penicillin_offline_g_l": "g/L",
    "biomass_offline_g_l": "g/L",
    "cer_g_h": "g/h",
    "ammonia_shots_kg": "kg",
    "viscosity_offline_cp": "cP",
}

# Only explicit process measurements appear here. Identifiers and ground truth do not.
INITIAL_MODEL_SENSORS = (
    "dissolved_oxygen_mg_l",
    "ph",
    "temperature_k",
    "pressure_bar",
)

EXTENDED_MODEL_SENSORS = (
    *INITIAL_MODEL_SENSORS,
    "aeration_lph",
    "substrate_feed_lph",
    "base_flow_lph",
    "coolant_flow_lph",
)

