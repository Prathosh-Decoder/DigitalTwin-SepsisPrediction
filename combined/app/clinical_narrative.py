"""Guarded narrative generation for the research dashboard."""
import json
import os
from functools import lru_cache


def _fmt(value, suffix=""):
    return "not recorded" if value is None else f"{value:.1f}{suffix}"


def _rules(payload):
    active = payload["active_alert"]
    forecast = payload["forecast"]
    abnormal = [v for v in payload["vitals"] if v["abnormal"] and v["value"] is not None]
    drivers = [d for d in forecast.get("drivers", []) if d.get("direction") == "up"]
    evidence = []
    for item in abnormal[:2]:
        evidence.append(f"{item['label']} {_fmt(item['value'], ' ' + item['unit'])}")
    for driver in drivers:
        if len(evidence) >= 2:
            break
        if driver["label"].lower() not in " ".join(evidence).lower():
            evidence.append(driver["label"])
    evidence = evidence[:2]
    evidence_text = " with " + " and ".join(evidence) if evidence else ""
    probability = forecast.get("probability")
    if active["alert"]:
        observation = (
            f"The active-alert model is positive and the six-hour sepsis forecast is "
            f"{_fmt(probability * 100 if probability is not None else None, '%')}{evidence_text}."
        )
        recommendation = "Promptly verify the measurements and escalate for clinician review under the local sepsis pathway."
    elif forecast["alert"]:
        observation = (
            f"No current alert is active, but estimated sepsis risk in the next six hours is "
            f"{_fmt(probability * 100 if probability is not None else None, '%')}{evidence_text}."
        )
        recommendation = "Repeat and validate abnormal observations, review the trend, and notify the responsible clinician if risk persists."
    elif payload["state"] == "WATCH":
        observation = f"The patient remains below both alert thresholds but has a watch-level risk pattern{evidence_text}."
        recommendation = "Continue scheduled monitoring and reassess when new vital signs or laboratory results arrive."
    else:
        observation = "Both models remain below their alert thresholds and no high-priority deterioration signal is present."
        recommendation = "Continue routine monitoring and apply normal clinical assessment when new data arrive."
    return {"observation": observation, "recommendation": recommendation, "source": "rules"}


def _llm_input(payload):
    return {
        "state": payload["state"],
        "icu_hour": payload["hour"],
        "active_alert": {
            "alert": payload["active_alert"]["alert"],
            "probability": payload["active_alert"]["probability"],
            "criticality_percentile": payload["active_alert"]["criticality"],
            "trend": payload["active_alert"]["trend"],
        },
        "six_hour_forecast": {
            "alert": payload["forecast"]["alert"],
            "probability": payload["forecast"]["probability"],
            "threshold": payload["forecast"]["threshold"],
            "trend": payload["forecast"]["trend"],
        },
        "vitals": payload["vitals"],
        "local_shap_drivers": payload["forecast"].get("drivers", []),
    }


@lru_cache(maxsize=256)
def _openai_narrative(serialized):
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            "You summarize an ICU research dashboard. Return exactly two concise sentences as JSON: "
            "one observation grounded only in the supplied measurements/model outputs, and one action-oriented "
            "recommendation limited to measurement verification, monitoring, and clinician review. Do not diagnose, "
            "prescribe treatment, or imply SHAP is causal. Do not mention hidden labels."
        ),
        input=serialized,
        text={
            "format": {
                "type": "json_schema",
                "name": "clinical_dashboard_narrative",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["observation", "recommendation"],
                    "additionalProperties": False,
                },
            }
        },
    )
    result = json.loads(response.output_text)
    result["source"] = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-5-mini')}"
    return result


def build_narrative(payload):
    """Use the LLM when configured, otherwise return a deterministic safe summary."""
    if not os.environ.get("OPENAI_API_KEY"):
        return _rules(payload)
    try:
        return _openai_narrative(json.dumps(_llm_input(payload), sort_keys=True, default=str))
    except Exception:
        fallback = _rules(payload)
        fallback["source"] = "rules-fallback"
        return fallback
