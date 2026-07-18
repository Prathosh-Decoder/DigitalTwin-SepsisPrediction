from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
MEASUREMENT_COLUMNS = VITALS + LABS
RAW_COLUMNS = MEASUREMENT_COLUMNS + DEMOGRAPHICS
LABEL_COLUMN = "SepsisLabel"


@dataclass(frozen=True)
class FeatureConfig:
    rolling_windows: tuple[int, ...] = (3, 6)
    delta_hours: tuple[int, ...] = (1, 3)
    include_clinical_features: bool = False
    include_lab_dynamics: bool = False
    include_targeted_features: bool = False
    include_advanced_features: bool = False
    include_physiology_trends: bool = False
    include_lab_trajectories: bool = False
    include_patient_context: bool = False
    include_literature_interactions: bool = False


DEFAULT_CONFIG = FeatureConfig()
LAB_DYNAMICS_CONFIG = FeatureConfig(include_lab_dynamics=True)
PHYSIOLOGY_TREND_CONFIG = FeatureConfig(include_physiology_trends=True)
LAB_TRAJECTORY_CONFIG = FeatureConfig(include_lab_trajectories=True)
LITERATURE_CORE_CONFIG = FeatureConfig(
    include_physiology_trends=True,
    include_lab_trajectories=True,
)
LITERATURE_TREND_CONFIG = FeatureConfig(
    include_physiology_trends=True,
    include_lab_trajectories=True,
    include_patient_context=True,
    include_literature_interactions=True,
)
ENHANCED_CONFIG = FeatureConfig(
    rolling_windows=(3, 6, 12, 24),
    delta_hours=(1, 3, 6, 12),
    include_clinical_features=True,
)
TARGETED_ADVANCED_CONFIG = FeatureConfig(
    rolling_windows=(3, 6, 12, 24),
    delta_hours=(1, 3, 6, 12),
    include_clinical_features=True,
    include_targeted_features=True,
)
ADVANCED_CONFIG = FeatureConfig(
    rolling_windows=(3, 6, 12, 24),
    delta_hours=(1, 3, 6, 12),
    include_clinical_features=True,
    include_targeted_features=True,
    include_advanced_features=True,
)


CLINICAL_FEATURES = [
    "shock_index",
    "pulse_pressure",
    "map_minus_dbp",
    "spo2_fio2_ratio",
    "bun_creatinine_ratio",
    "qsofa_sbp_low",
    "qsofa_resp_high",
    "sirs_hr_high",
    "sirs_resp_high",
    "sirs_temp_abnormal",
    "wbc_abnormal",
    "lactate_high",
    "age_over_65",
]

LAB_DYNAMICS_LABS = [
    "Lactate",
    "WBC",
    "BUN",
    "Creatinine",
    "Platelets",
    "Bilirubin_total",
    "pH",
    "PaCO2",
    "HCO3",
    "BaseExcess",
    "FiO2",
]

LAB_DYNAMICS_FEATURES = (
    [f"{column}_lab_abnormal" for column in LAB_DYNAMICS_LABS]
    + [f"{column}_since_abnormal" for column in LAB_DYNAMICS_LABS]
    + [f"{column}_observed_delta" for column in LAB_DYNAMICS_LABS]
    + [f"{column}_observed_pct_delta" for column in LAB_DYNAMICS_LABS]
    + [f"{column}_observed_min_3" for column in LAB_DYNAMICS_LABS]
    + [f"{column}_observed_max_3" for column in LAB_DYNAMICS_LABS]
    + [
        "lab_panel_measured_count",
        "lab_panel_measured_count_6h",
        "lab_panel_measured_count_24h",
        "new_lab_panel_indicator",
        "abnormal_lab_count",
        "abnormal_lab_count_24h",
        "renal_lab_score",
        "liver_lab_score",
        "coagulation_lab_score",
        "acid_base_lab_score",
        "oxygenation_lab_score",
        "organ_lab_score",
    ]
)

PHYSIOLOGY_TREND_VITALS = ["HR", "MAP", "SBP", "Resp", "Temp", "O2Sat"]
PHYSIOLOGY_TREND_WINDOWS = (6, 12)
PHYSIOLOGY_TREND_FEATURES = (
    [
        f"{column}_{stat}_{window}h"
        for column in PHYSIOLOGY_TREND_VITALS
        for window in PHYSIOLOGY_TREND_WINDOWS
        for stat in ("std", "range", "slope", "direction_changes")
    ]
    + [
        "hr_high_run_length",
        "map_low_run_length",
        "sbp_low_run_length",
        "resp_high_run_length",
        "temp_abnormal_run_length",
        "o2sat_low_run_length",
    ]
)

LAB_TRAJECTORY_LABS = ["PTT", "WBC", "Platelets", "Lactate", "Creatinine", "BUN"]
LAB_TRAJECTORY_WINDOWS = (12, 24)
LAB_TRAJECTORY_FEATURES = (
    [
        f"{column}_observed_{stat}_{window}h"
        for column in LAB_TRAJECTORY_LABS
        for window in LAB_TRAJECTORY_WINDOWS
        for stat in ("min", "max", "range")
    ]
    + [f"{column}_observed_change" for column in LAB_TRAJECTORY_LABS]
    + [f"{column}_observed_velocity" for column in LAB_TRAJECTORY_LABS]
)

PATIENT_CONTEXT_COLUMNS = PHYSIOLOGY_TREND_VITALS + LAB_TRAJECTORY_LABS
PATIENT_CONTEXT_FEATURES = (
    [f"{column}_from_first" for column in PATIENT_CONTEXT_COLUMNS]
    + [f"{column}_pct_from_first" for column in PATIENT_CONTEXT_COLUMNS]
    + [
        "vital_measurement_frequency",
        "lab_measurement_frequency",
        "target_lab_measurement_frequency",
        "recent_to_cumulative_lab_intensity",
    ]
)

LITERATURE_INTERACTION_FEATURES = [
    "literature_shock_index",
    "literature_spo2_fio2_ratio",
    "map_lactate_burden",
    "ptt_platelet_burden",
    "temp_resp_burden",
    "renal_burden",
    "multi_organ_interaction_count",
]

ADVANCED_FEATURES = [
    "vital_measurement_intensity",
    "key_lab_measurement_intensity",
    "fio2_measured_count_6h",
    "lactate_measured_count_24h",
    "wbc_measured_count_24h",
    "bun_measured_count_24h",
    "creatinine_measured_count_24h",
    "qsofa_score",
    "sirs_score",
    "organ_dysfunction_score",
    "map_low_count_6h",
    "hr_high_count_6h",
    "resp_high_count_6h",
    "spo2_low_count_6h",
    "temp_abnormal_count_6h",
    "map_low_count_24h",
    "hr_high_count_24h",
    "resp_high_count_24h",
    "spo2_low_count_24h",
    "temp_abnormal_count_24h",
    "map_drop_6h",
    "spo2_drop_6h",
    "resp_rise_6h",
    "hr_rise_6h",
]

BROAD_ADVANCED_FEATURES = [
    "vitals_measured_count",
    "labs_measured_count",
    "measurements_measured_count",
    "vitals_measured_count_6h",
    "labs_measured_count_6h",
    "measurements_measured_count_6h",
    "vitals_measured_count_24h",
    "labs_measured_count_24h",
    "measurements_measured_count_24h",
    "hypotension_map_low",
    "hypotension_sbp_low",
    "tachycardia",
    "tachypnea",
    "hypoxemia",
    "fever",
    "hypothermia",
    "renal_abnormal",
    "platelets_low",
    "bilirubin_high",
    "ph_abnormal",
    "paco2_high",
    "sbp_low_count_6h",
    "sbp_low_count_24h",
    "sbp_drop_6h",
]

FEATURE_SUFFIXES = [
    "_missing",
    "_ffill",
    "_since_measured",
    "_delta_1h",
    "_delta_3h",
    "_delta_6h",
    "_delta_12h",
    "_mean_3h",
    "_mean_6h",
    "_mean_12h",
    "_mean_24h",
    "_min_3h",
    "_min_6h",
    "_min_12h",
    "_min_24h",
    "_max_3h",
    "_max_6h",
    "_max_12h",
    "_max_24h",
]

SHAP_COMPACT_GROUPS = {
    "ICULOS",
    "icu_hour_log1p",
    "HospAdmTime",
    "hospital_to_icu_lag_hours",
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HR",
    "Resp",
    "Temp",
    "MAP",
    "SBP",
    "DBP",
    "O2Sat",
    "FiO2",
    "Lactate",
    "WBC",
    "BUN",
    "Creatinine",
    "PTT",
    "AST",
    "Platelets",
    "Glucose",
    "PaCO2",
    "SaO2",
    "EtCO2",
    "shock_index",
    "spo2_fio2_ratio",
    "bun_creatinine_ratio",
    "pulse_pressure",
    "map_minus_dbp",
    "sirs_temp_abnormal",
}


def expected_feature_names(config: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    names: list[str] = []
    names.extend(RAW_COLUMNS)
    names.extend([f"{column}_missing" for column in MEASUREMENT_COLUMNS])
    names.extend([f"{column}_ffill" for column in MEASUREMENT_COLUMNS])
    names.extend([f"{column}_since_measured" for column in MEASUREMENT_COLUMNS])
    for column in VITALS:
        for delta_hour in config.delta_hours:
            names.append(f"{column}_delta_{delta_hour}h")
        for window in config.rolling_windows:
            names.extend(
                [
                    f"{column}_mean_{window}h",
                    f"{column}_min_{window}h",
                    f"{column}_max_{window}h",
                ]
            )
    names.extend(["icu_hour_log1p", "hospital_to_icu_lag_hours"])
    if config.include_clinical_features:
        names.extend(CLINICAL_FEATURES)
    if config.include_lab_dynamics:
        names.extend(LAB_DYNAMICS_FEATURES)
    if config.include_targeted_features:
        names.extend(ADVANCED_FEATURES)
    if config.include_advanced_features:
        names.extend(BROAD_ADVANCED_FEATURES)
    if config.include_physiology_trends:
        names.extend(PHYSIOLOGY_TREND_FEATURES)
    if config.include_lab_trajectories:
        names.extend(LAB_TRAJECTORY_FEATURES)
    if config.include_patient_context:
        names.extend(PATIENT_CONTEXT_FEATURES)
    if config.include_literature_interactions:
        names.extend(LITERATURE_INTERACTION_FEATURES)
    return names


def feature_group_name(feature_name: str) -> str:
    for suffix in FEATURE_SUFFIXES:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def compact_feature_names(config: FeatureConfig = ENHANCED_CONFIG) -> list[str]:
    return [
        feature_name
        for feature_name in expected_feature_names(config)
        if feature_group_name(feature_name) in SHAP_COMPACT_GROUPS
    ]


def feature_config_from_dict(payload: dict[str, Any] | FeatureConfig | None) -> FeatureConfig:
    if payload is None:
        return DEFAULT_CONFIG
    if isinstance(payload, FeatureConfig):
        return payload
    return FeatureConfig(
        rolling_windows=tuple(payload.get("rolling_windows", DEFAULT_CONFIG.rolling_windows)),
        delta_hours=tuple(payload.get("delta_hours", DEFAULT_CONFIG.delta_hours)),
        include_clinical_features=bool(payload.get("include_clinical_features", False)),
        include_lab_dynamics=bool(payload.get("include_lab_dynamics", False)),
        include_targeted_features=bool(
            payload.get("include_targeted_features", payload.get("include_advanced_features", False))
        ),
        include_advanced_features=bool(payload.get("include_advanced_features", False)),
        include_physiology_trends=bool(payload.get("include_physiology_trends", False)),
        include_lab_trajectories=bool(payload.get("include_lab_trajectories", False)),
        include_patient_context=bool(payload.get("include_patient_context", False)),
        include_literature_interactions=bool(payload.get("include_literature_interactions", False)),
    )


def _ensure_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()
    for column in RAW_COLUMNS + [LABEL_COLUMN]:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared


def _numeric_frame(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return dataframe[columns].apply(pd.to_numeric, errors="coerce")


def _hours_since_measured(frame: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {}
    index = np.arange(len(frame))
    for column in frame.columns:
        observed = frame[column].notna().to_numpy()
        last_observed = np.where(observed, index, np.nan)
        last_observed = pd.Series(last_observed).ffill().to_numpy()
        values[f"{column}_since_measured"] = index - last_observed
    return pd.DataFrame(values, index=frame.index)


def _hours_since_true(frame: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {}
    index = np.arange(len(frame))
    for column in frame.columns:
        observed = frame[column].fillna(False).astype(bool).to_numpy()
        last_observed = np.where(observed, index, np.nan)
        last_observed = pd.Series(last_observed).ffill().to_numpy()
        values[f"{column}_since_abnormal"] = index - last_observed
    return pd.DataFrame(values, index=frame.index)


def _rolling_observed_stat(series: pd.Series, window: int, stat: str) -> pd.Series:
    observed = series.dropna()
    if observed.empty:
        return pd.Series(np.nan, index=series.index)
    rolling = observed.rolling(window=window, min_periods=1)
    if stat == "min":
        values = rolling.min()
    elif stat == "max":
        values = rolling.max()
    else:
        raise ValueError(f"Unsupported observed rolling stat: {stat}")
    return values.reindex(series.index).ffill()


def _consecutive_true_hours(flag: pd.Series) -> pd.Series:
    values = flag.fillna(False).astype(bool).to_numpy()
    counts = np.zeros(len(values), dtype=float)
    running = 0
    for index, value in enumerate(values):
        running = running + 1 if value else 0
        counts[index] = running
    return pd.Series(counts, index=flag.index)


def _first_observed_value(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    observed_positions = np.flatnonzero(np.isfinite(values))
    result = np.full(len(values), np.nan, dtype=float)
    if observed_positions.size:
        first_position = int(observed_positions[0])
        result[first_position:] = values[first_position]
    return pd.Series(result, index=series.index)


def _observed_change_and_velocity(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    observed_positions = np.flatnonzero(np.isfinite(values))
    change = np.full(len(values), np.nan, dtype=float)
    velocity = np.full(len(values), np.nan, dtype=float)
    if observed_positions.size >= 2:
        value_changes = np.diff(values[observed_positions])
        elapsed_hours = np.diff(observed_positions).astype(float)
        change[observed_positions[1:]] = value_changes
        velocity[observed_positions[1:]] = value_changes / elapsed_hours
    return (
        pd.Series(change, index=series.index).ffill(),
        pd.Series(velocity, index=series.index).ffill(),
    )


def build_feature_frame(
    patient_dataframe: pd.DataFrame,
    config: FeatureConfig = DEFAULT_CONFIG,
    include_label: bool = True,
) -> pd.DataFrame:
    """Build causal patient-hour features from one patient's time series.

    All transformations use values from the current or previous ICU hours only.
    This avoids future-data leakage during training and live inference.
    """
    dataframe = _ensure_columns(patient_dataframe)
    raw = _numeric_frame(dataframe, RAW_COLUMNS)
    measurements = raw[MEASUREMENT_COLUMNS]
    ffill = measurements.ffill()

    parts = [
        raw,
        measurements.isna().astype(np.int8).rename(columns=lambda column: f"{column}_missing"),
        ffill.rename(columns=lambda column: f"{column}_ffill"),
        _hours_since_measured(measurements),
    ]

    vital_features: dict[str, pd.Series] = {}
    for column in VITALS:
        series = ffill[column]
        for delta_hour in config.delta_hours:
            vital_features[f"{column}_delta_{delta_hour}h"] = series.diff(delta_hour)
        for window in config.rolling_windows:
            rolling = series.rolling(window=window, min_periods=1)
            vital_features[f"{column}_mean_{window}h"] = rolling.mean()
            vital_features[f"{column}_min_{window}h"] = rolling.min()
            vital_features[f"{column}_max_{window}h"] = rolling.max()
    parts.append(pd.DataFrame(vital_features, index=dataframe.index))

    derived = pd.DataFrame(index=dataframe.index)
    derived["icu_hour_log1p"] = np.log1p(raw["ICULOS"].clip(lower=0))
    derived["hospital_to_icu_lag_hours"] = raw["HospAdmTime"].abs()
    if config.include_clinical_features:
        safe_sbp = ffill["SBP"].replace(0, np.nan)
        safe_dbp = ffill["DBP"].replace(0, np.nan)
        safe_fio2 = ffill["FiO2"].replace(0, np.nan)
        safe_creatinine = ffill["Creatinine"].replace(0, np.nan)
        derived["shock_index"] = ffill["HR"] / safe_sbp
        derived["pulse_pressure"] = ffill["SBP"] - ffill["DBP"]
        derived["map_minus_dbp"] = ffill["MAP"] - ffill["DBP"]
        derived["spo2_fio2_ratio"] = ffill["O2Sat"] / safe_fio2
        derived["bun_creatinine_ratio"] = ffill["BUN"] / safe_creatinine
        derived["qsofa_sbp_low"] = (ffill["SBP"] <= 100).astype(float)
        derived["qsofa_resp_high"] = (ffill["Resp"] >= 22).astype(float)
        derived["sirs_hr_high"] = (ffill["HR"] > 90).astype(float)
        derived["sirs_resp_high"] = (ffill["Resp"] > 20).astype(float)
        derived["sirs_temp_abnormal"] = ((ffill["Temp"] > 38) | (ffill["Temp"] < 36)).astype(float)
        derived["wbc_abnormal"] = ((ffill["WBC"] > 12) | (ffill["WBC"] < 4)).astype(float)
        derived["lactate_high"] = (ffill["Lactate"] >= 2).astype(float)
        derived["age_over_65"] = (raw["Age"] >= 65).astype(float)
    if config.include_lab_dynamics:
        lab_abnormal = pd.DataFrame(index=dataframe.index)
        lab_abnormal["Lactate"] = ffill["Lactate"] >= 2.0
        lab_abnormal["WBC"] = (ffill["WBC"] > 12.0) | (ffill["WBC"] < 4.0)
        lab_abnormal["BUN"] = ffill["BUN"] >= 28.0
        lab_abnormal["Creatinine"] = ffill["Creatinine"] >= 2.0
        lab_abnormal["Platelets"] = ffill["Platelets"] < 150.0
        lab_abnormal["Bilirubin_total"] = ffill["Bilirubin_total"] >= 2.0
        lab_abnormal["pH"] = (ffill["pH"] < 7.35) | (ffill["pH"] > 7.45)
        lab_abnormal["PaCO2"] = ffill["PaCO2"] > 45.0
        lab_abnormal["HCO3"] = ffill["HCO3"] < 22.0
        lab_abnormal["BaseExcess"] = ffill["BaseExcess"] < -2.0
        lab_abnormal["FiO2"] = ffill["FiO2"] >= 0.5

        for column in LAB_DYNAMICS_LABS:
            derived[f"{column}_lab_abnormal"] = lab_abnormal[column].astype(float)

            observed_series = measurements[column]
            observed_delta = observed_series.dropna().diff().reindex(dataframe.index).ffill()
            previous_observed = observed_series.dropna().shift(1).replace(0, np.nan)
            observed_pct_delta = (observed_series.dropna().diff() / previous_observed.abs()).reindex(
                dataframe.index
            ).ffill()
            derived[f"{column}_observed_delta"] = observed_delta
            derived[f"{column}_observed_pct_delta"] = observed_pct_delta
            derived[f"{column}_observed_min_3"] = _rolling_observed_stat(observed_series, window=3, stat="min")
            derived[f"{column}_observed_max_3"] = _rolling_observed_stat(observed_series, window=3, stat="max")

        since_abnormal = _hours_since_true(lab_abnormal).rename(
            columns={f"{column}_since_abnormal": f"{column}_since_abnormal" for column in LAB_DYNAMICS_LABS}
        )
        parts.append(since_abnormal)

        lab_observed = measurements[LAB_DYNAMICS_LABS].notna().astype(float)
        lab_panel_count = lab_observed.sum(axis=1)
        abnormal_count = lab_abnormal.astype(float).sum(axis=1)
        renal_score = lab_abnormal[["BUN", "Creatinine"]].astype(float).sum(axis=1)
        liver_score = lab_abnormal[["Bilirubin_total"]].astype(float).sum(axis=1)
        coagulation_score = lab_abnormal[["Platelets"]].astype(float).sum(axis=1)
        acid_base_score = lab_abnormal[["pH", "PaCO2", "HCO3", "BaseExcess", "Lactate"]].astype(float).sum(axis=1)
        oxygenation_score = lab_abnormal[["FiO2"]].astype(float).sum(axis=1)

        derived["lab_panel_measured_count"] = lab_panel_count
        derived["lab_panel_measured_count_6h"] = lab_panel_count.rolling(window=6, min_periods=1).sum()
        derived["lab_panel_measured_count_24h"] = lab_panel_count.rolling(window=24, min_periods=1).sum()
        derived["new_lab_panel_indicator"] = (lab_panel_count >= 3).astype(float)
        derived["abnormal_lab_count"] = abnormal_count
        derived["abnormal_lab_count_24h"] = abnormal_count.rolling(window=24, min_periods=1).sum()
        derived["renal_lab_score"] = renal_score
        derived["liver_lab_score"] = liver_score
        derived["coagulation_lab_score"] = coagulation_score
        derived["acid_base_lab_score"] = acid_base_score
        derived["oxygenation_lab_score"] = oxygenation_score
        derived["organ_lab_score"] = (
            (renal_score > 0).astype(float)
            + (liver_score > 0).astype(float)
            + (coagulation_score > 0).astype(float)
            + (acid_base_score > 0).astype(float)
            + (oxygenation_score > 0).astype(float)
        )
    if config.include_targeted_features or config.include_advanced_features:
        observed = measurements.notna().astype(float)
        vital_observed = observed[VITALS]
        lab_observed = observed[LABS]
        key_labs = ["Lactate", "WBC", "BUN", "Creatinine"]
        derived["vital_measurement_intensity"] = vital_observed[["HR", "O2Sat", "Temp", "MAP", "Resp"]].sum(axis=1)
        derived["key_lab_measurement_intensity"] = lab_observed[key_labs].sum(axis=1)
        derived["fio2_measured_count_6h"] = observed["FiO2"].rolling(window=6, min_periods=1).sum()
        for column in key_labs:
            derived[f"{column.lower()}_measured_count_24h"] = observed[column].rolling(window=24, min_periods=1).sum()

        hypotension_map_low = (ffill["MAP"] < 65).astype(float)
        hypotension_sbp_low = (ffill["SBP"] < 90).astype(float)
        tachycardia = (ffill["HR"] > 100).astype(float)
        tachypnea = (ffill["Resp"] >= 22).astype(float)
        hypoxemia = (ffill["O2Sat"] < 92).astype(float)
        fever = (ffill["Temp"] > 38).astype(float)
        hypothermia = (ffill["Temp"] < 36).astype(float)
        temp_abnormal = ((ffill["Temp"] > 38) | (ffill["Temp"] < 36)).astype(float)
        renal_abnormal = ((ffill["Creatinine"] >= 2.0) | (ffill["BUN"] >= 28)).astype(float)
        platelets_low = (ffill["Platelets"] < 150).astype(float)
        bilirubin_high = (ffill["Bilirubin_total"] >= 2.0).astype(float)
        ph_abnormal = ((ffill["pH"] < 7.35) | (ffill["pH"] > 7.45)).astype(float)
        paco2_high = (ffill["PaCO2"] > 45).astype(float)

        derived["qsofa_score"] = (
            (ffill["SBP"] <= 100).astype(float)
            + (ffill["Resp"] >= 22).astype(float)
        )
        derived["sirs_score"] = (
            tachycardia
            + (ffill["Resp"] > 20).astype(float)
            + temp_abnormal
            + ((ffill["WBC"] > 12) | (ffill["WBC"] < 4)).astype(float)
        )
        derived["organ_dysfunction_score"] = (
            hypotension_map_low
            + hypoxemia
            + renal_abnormal
            + platelets_low
            + bilirubin_high
            + (ffill["Lactate"] >= 2).astype(float)
        )

        count_sources = {
            "map_low": hypotension_map_low,
            "hr_high": tachycardia,
            "resp_high": tachypnea,
            "spo2_low": hypoxemia,
            "temp_abnormal": temp_abnormal,
        }
        for window in (6, 24):
            for name, flag in count_sources.items():
                derived[f"{name}_count_{window}h"] = flag.rolling(window=window, min_periods=1).sum()

        derived["map_drop_6h"] = (ffill["MAP"].shift(6) - ffill["MAP"]).clip(lower=0)
        derived["spo2_drop_6h"] = (ffill["O2Sat"].shift(6) - ffill["O2Sat"]).clip(lower=0)
        derived["resp_rise_6h"] = (ffill["Resp"] - ffill["Resp"].shift(6)).clip(lower=0)
        derived["hr_rise_6h"] = (ffill["HR"] - ffill["HR"].shift(6)).clip(lower=0)
        if config.include_advanced_features:
            derived["vitals_measured_count"] = vital_observed.sum(axis=1)
            derived["labs_measured_count"] = lab_observed.sum(axis=1)
            derived["measurements_measured_count"] = observed.sum(axis=1)
            for window in (6, 24):
                derived[f"vitals_measured_count_{window}h"] = vital_observed.sum(axis=1).rolling(
                    window=window, min_periods=1
                ).sum()
                derived[f"labs_measured_count_{window}h"] = lab_observed.sum(axis=1).rolling(
                    window=window, min_periods=1
                ).sum()
                derived[f"measurements_measured_count_{window}h"] = observed.sum(axis=1).rolling(
                    window=window, min_periods=1
                ).sum()
            broad_flags = {
                "hypotension_map_low": hypotension_map_low,
                "hypotension_sbp_low": hypotension_sbp_low,
                "tachycardia": tachycardia,
                "tachypnea": tachypnea,
                "hypoxemia": hypoxemia,
                "fever": fever,
                "hypothermia": hypothermia,
                "renal_abnormal": renal_abnormal,
                "platelets_low": platelets_low,
                "bilirubin_high": bilirubin_high,
                "ph_abnormal": ph_abnormal,
                "paco2_high": paco2_high,
            }
            for name, flag in broad_flags.items():
                derived[name] = flag
            for window in (6, 24):
                derived[f"sbp_low_count_{window}h"] = hypotension_sbp_low.rolling(window=window, min_periods=1).sum()
            derived["sbp_drop_6h"] = (ffill["SBP"].shift(6) - ffill["SBP"]).clip(lower=0)
    if config.include_physiology_trends:
        trend_features: dict[str, pd.Series] = {}
        for column in PHYSIOLOGY_TREND_VITALS:
            series = ffill[column]
            difference = series.diff()
            direction = np.sign(difference)
            direction_change = ((direction * direction.shift(1)) < 0).astype(float)
            for window in PHYSIOLOGY_TREND_WINDOWS:
                rolling = series.rolling(window=window, min_periods=2)
                trend_features[f"{column}_std_{window}h"] = rolling.std(ddof=0)
                trend_features[f"{column}_range_{window}h"] = rolling.max() - rolling.min()
                trend_features[f"{column}_slope_{window}h"] = (
                    series - series.shift(window - 1)
                ) / float(window - 1)
                trend_features[f"{column}_direction_changes_{window}h"] = direction_change.rolling(
                    window=window, min_periods=1
                ).sum()

        trend_features["hr_high_run_length"] = _consecutive_true_hours(ffill["HR"] > 100)
        trend_features["map_low_run_length"] = _consecutive_true_hours(ffill["MAP"] < 65)
        trend_features["sbp_low_run_length"] = _consecutive_true_hours(ffill["SBP"] < 90)
        trend_features["resp_high_run_length"] = _consecutive_true_hours(ffill["Resp"] >= 22)
        trend_features["temp_abnormal_run_length"] = _consecutive_true_hours(
            (ffill["Temp"] > 38) | (ffill["Temp"] < 36)
        )
        trend_features["o2sat_low_run_length"] = _consecutive_true_hours(ffill["O2Sat"] < 92)
        parts.append(pd.DataFrame(trend_features, index=dataframe.index))

    if config.include_lab_trajectories:
        lab_trajectory_features: dict[str, pd.Series] = {}
        for column in LAB_TRAJECTORY_LABS:
            observed_series = measurements[column]
            for window in LAB_TRAJECTORY_WINDOWS:
                rolling = observed_series.rolling(window=window, min_periods=1)
                rolling_min = rolling.min()
                rolling_max = rolling.max()
                lab_trajectory_features[f"{column}_observed_min_{window}h"] = rolling_min
                lab_trajectory_features[f"{column}_observed_max_{window}h"] = rolling_max
                lab_trajectory_features[f"{column}_observed_range_{window}h"] = rolling_max - rolling_min
            change, velocity = _observed_change_and_velocity(observed_series)
            lab_trajectory_features[f"{column}_observed_change"] = change
            lab_trajectory_features[f"{column}_observed_velocity"] = velocity
        parts.append(pd.DataFrame(lab_trajectory_features, index=dataframe.index))

    if config.include_patient_context:
        patient_context_features: dict[str, pd.Series] = {}
        for column in PATIENT_CONTEXT_COLUMNS:
            first_value = _first_observed_value(measurements[column])
            current_value = ffill[column]
            patient_context_features[f"{column}_from_first"] = current_value - first_value
            patient_context_features[f"{column}_pct_from_first"] = (
                current_value - first_value
            ) / first_value.abs().replace(0, np.nan)

        elapsed_hours = pd.Series(np.arange(1, len(dataframe) + 1, dtype=float), index=dataframe.index)
        vital_counts = measurements[PHYSIOLOGY_TREND_VITALS].notna().sum(axis=1)
        lab_counts = measurements[LABS].notna().sum(axis=1)
        target_lab_counts = measurements[LAB_TRAJECTORY_LABS].notna().sum(axis=1)
        patient_context_features["vital_measurement_frequency"] = (
            vital_counts.cumsum() / elapsed_hours / len(PHYSIOLOGY_TREND_VITALS)
        )
        patient_context_features["lab_measurement_frequency"] = lab_counts.cumsum() / elapsed_hours / len(LABS)
        patient_context_features["target_lab_measurement_frequency"] = (
            target_lab_counts.cumsum() / elapsed_hours / len(LAB_TRAJECTORY_LABS)
        )
        cumulative_hourly_lab_intensity = target_lab_counts.cumsum() / elapsed_hours
        recent_lab_intensity = target_lab_counts.rolling(window=6, min_periods=1).mean()
        patient_context_features["recent_to_cumulative_lab_intensity"] = (
            recent_lab_intensity / cumulative_hourly_lab_intensity.replace(0, np.nan)
        )
        parts.append(pd.DataFrame(patient_context_features, index=dataframe.index))

    if config.include_literature_interactions:
        safe_sbp = ffill["SBP"].replace(0, np.nan)
        safe_fio2 = ffill["FiO2"].replace(0, np.nan)
        map_burden = (65 - ffill["MAP"]).clip(lower=0)
        lactate_burden = (ffill["Lactate"] - 2).clip(lower=0)
        ptt_burden = (ffill["PTT"] - 35).clip(lower=0)
        platelet_burden = (150 - ffill["Platelets"]).clip(lower=0)
        renal_creatinine = (ffill["Creatinine"] - 1.2).clip(lower=0)
        renal_bun = (ffill["BUN"] - 20).clip(lower=0)
        interaction_features = {
            "literature_shock_index": ffill["HR"] / safe_sbp,
            "literature_spo2_fio2_ratio": ffill["O2Sat"] / safe_fio2,
            "map_lactate_burden": map_burden * lactate_burden,
            "ptt_platelet_burden": ptt_burden * platelet_burden,
            "temp_resp_burden": (ffill["Temp"] - 37).abs() * (ffill["Resp"] - 20).clip(lower=0),
            "renal_burden": renal_creatinine * renal_bun,
            "multi_organ_interaction_count": (
                (map_burden > 0).astype(float)
                + (lactate_burden > 0).astype(float)
                + (platelet_burden > 0).astype(float)
                + (renal_creatinine > 0).astype(float)
                + (ffill["O2Sat"] < 92).astype(float)
            ),
        }
        parts.append(pd.DataFrame(interaction_features, index=dataframe.index))
    parts.append(derived)

    features = pd.concat(parts, axis=1)
    features = features.reindex(columns=expected_feature_names(config))

    if include_label:
        features[LABEL_COLUMN] = pd.to_numeric(dataframe[LABEL_COLUMN], errors="coerce").fillna(0).astype(np.int8)
    return features


def build_latest_feature_row(patient_records: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(patient_records)
    features = build_feature_frame(dataframe, include_label=False)
    return features.tail(1)
