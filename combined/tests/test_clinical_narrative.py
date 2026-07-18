import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from clinical_narrative import build_narrative  # noqa: E402


def payload(state="FORECAST", active=False, forecast=True):
    return {
        "state": state,
        "hour": 24,
        "active_alert": {"alert": active, "probability": 0.18, "criticality": 82, "trend": "rising"},
        "forecast": {
            "alert": forecast,
            "probability": 0.42,
            "threshold": 0.31,
            "trend": "rising",
            "drivers": [{"label": "latest lactate", "direction": "up"}],
        },
        "vitals": [{"label": "MAP", "value": 61.0, "unit": "mmHg", "abnormal": True}],
    }


class NarrativeTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_rules_fallback_has_two_guarded_sentences(self):
        result = build_narrative(payload())
        self.assertEqual(result["source"], "rules")
        self.assertTrue(result["observation"].endswith("."))
        self.assertTrue(result["recommendation"].endswith("."))
        self.assertNotIn("diagnos", (result["observation"] + result["recommendation"]).lower())

    @patch.dict("os.environ", {}, clear=True)
    def test_active_alert_takes_priority(self):
        result = build_narrative(payload(state="ACTIVE", active=True))
        self.assertIn("active-alert model is positive", result["observation"])
        self.assertIn("clinician review", result["recommendation"])


if __name__ == "__main__":
    unittest.main()
