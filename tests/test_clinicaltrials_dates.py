import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "clinicaltrials_dates.py"


def load_dates_module():
    spec = importlib.util.spec_from_file_location("clinicaltrials_dates", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClinicalTrialsDateExtractionTests(unittest.TestCase):
    def test_extract_trial_date_metadata_normalizes_precision_and_priority(self):
        module = load_dates_module()
        raw = {
            "protocolSection": {
                "statusModule": {
                    "primaryCompletionDateStruct": {"date": "June 2020"},
                    "completionDateStruct": {"date": "2021-04-03"},
                    "startDateStruct": {"date": "2019"},
                    "resultsFirstSubmitDate": "2022-01-15",
                }
            }
        }

        metadata = module.extract_trial_date_metadata(raw, nct_id="NCT123")

        self.assertEqual(metadata["nct_id"], "NCT123")
        self.assertEqual(metadata["primary_completion_date"], "2020-06-15")
        self.assertEqual(metadata["primary_completion_date_precision"], "month")
        self.assertEqual(metadata["completion_date"], "2021-04-03")
        self.assertEqual(metadata["completion_date_precision"], "day")
        self.assertEqual(metadata["start_date"], "2019-06-30")
        self.assertEqual(metadata["start_date_precision"], "year")
        self.assertEqual(metadata["results_first_posted_date"], "2022-01-15")
        self.assertEqual(metadata["temporal_sort_date"], "2020-06-15")
        self.assertEqual(metadata["temporal_sort_source"], "primary_completion_date")

    def test_date_missingness_report_counts_sources_and_precisions(self):
        module = load_dates_module()
        rows = [
            {
                "nct_id": "NCT1",
                "primary_completion_date": "2020-06-15",
                "primary_completion_date_precision": "month",
                "completion_date": None,
                "completion_date_precision": "missing",
                "start_date": "2019-06-30",
                "start_date_precision": "year",
                "temporal_sort_source": "primary_completion_date",
            },
            {
                "nct_id": "NCT2",
                "primary_completion_date": None,
                "primary_completion_date_precision": "missing",
                "completion_date": "2021-01-05",
                "completion_date_precision": "day",
                "start_date": None,
                "start_date_precision": "missing",
                "temporal_sort_source": "completion_date",
            },
        ]

        report = module.date_missingness_report(rows)

        self.assertEqual(report["trial_count"], 2)
        self.assertEqual(report["temporal_sort_sources"]["primary_completion_date"], 1)
        self.assertEqual(report["temporal_sort_sources"]["completion_date"], 1)
        self.assertEqual(report["field_missing_counts"]["primary_completion_date"], 1)
        self.assertEqual(report["precision_counts"]["primary_completion_date"]["month"], 1)
        self.assertEqual(report["precision_counts"]["completion_date"]["day"], 1)

    def test_extract_trial_date_metadata_supports_legacy_results_posted_record_dates(self):
        module = load_dates_module()
        raw = {
            "Study details": {"1. NCT number": "NCTLEGACY"},
            "Results Posted": {
                "4. Study Record Dates": {
                    "Study Start Date": "2017-03-17 (Actual)",
                    "Primary Completion Date": "2019-08-29 (Actual)",
                    "Study Completion Date": "2019-08-29 (Actual)",
                    "Results First Posted Date": "2022-06-29 (Actual)",
                }
            },
        }

        metadata = module.extract_trial_date_metadata(raw)

        self.assertEqual(metadata["nct_id"], "NCTLEGACY")
        self.assertEqual(metadata["primary_completion_date"], "2019-08-29")
        self.assertEqual(metadata["primary_completion_date_precision"], "day")
        self.assertEqual(metadata["completion_date"], "2019-08-29")
        self.assertEqual(metadata["results_first_posted_date"], "2022-06-29")
        self.assertEqual(metadata["start_date"], "2017-03-17")
        self.assertEqual(metadata["temporal_sort_source"], "primary_completion_date")


if __name__ == "__main__":
    unittest.main()
