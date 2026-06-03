from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_mixture_prior() -> Any:
    if __package__:
        try:
            from . import mixture_prior as package_mixture_prior

            return package_mixture_prior
        except ImportError:
            pass

    sibling_path = Path(__file__).with_name("mixture_prior.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "_oncology_trial_similarity_mixture_prior",
            sibling_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (ImportError, OSError):
        return None


mixture_prior = _load_mixture_prior()


DEFAULT_DB_ROOT = Path(
    "/Users/wang/PHD/clinic.gov/Oncology_All_Trials/Oncology_All_Trials"
)


SUMMARY_PROMPT_PATH = "oncology_trial_similarity_pipeline.md"


ASPECT_WEIGHTS = {
    "disease_population": 0.30,
    "intervention": 0.25,
    "endpoint": 0.20,
    "design": 0.15,
    "results_safety": 0.10,
}


DEFAULT_CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_RETRIEVAL_BACKEND = "clinicalbert"
RETRIEVAL_BACKENDS = ("clinicalbert", "trial2vec", "secret")


def ensure_supported_retrieval_backend(backend: str) -> None:
    if backend not in RETRIEVAL_BACKENDS:
        raise ValueError(
            f"Unsupported retrieval backend: {backend}. "
            f"Supported backends: {', '.join(RETRIEVAL_BACKENDS)}"
        )
    if backend == "secret":
        raise NotImplementedError(
            "SECRET retrieval is reserved for the protocol-summary backend and is not implemented in this revision."
        )


@dataclass
class TrialRecord:
    nct_id: str
    folder: Path
    json_path: Path
    protocol_path: Path | None
    sap_path: Path | None
    raw_json: dict[str, Any]
    extracted: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_trial_json(folder: Path) -> Path | None:
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        return None
    exact = folder / f"{folder.name}.json"
    return exact if exact.exists() else json_files[0]


def find_supporting_pdfs(folder: Path) -> dict[str, Path | None]:
    pdfs = sorted(folder.glob("*.pdf"))
    protocol = None
    sap = None
    for pdf in pdfs:
        name = pdf.name.lower()
        if protocol is None and "protocol" in name:
            protocol = pdf
        if sap is None and (
            "statistical_analysis" in name
            or "statistical analysis" in name
            or "sap" in name
        ):
            sap = pdf
    return {"protocol": protocol, "sap": sap}


def read_pdf_excerpt(path: Path | None, max_chars: int = 12000) -> str:
    if path is None or not path.exists():
        return ""
    if shutil.which("pdftotext") is None:
        return read_pdf_excerpt_with_python(path, max_chars=max_chars)
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-q", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return read_pdf_excerpt_with_python(path, max_chars=max_chars)
    return clean_text(result.stdout[:max_chars])


def read_pdf_excerpt_with_python(path: Path, max_chars: int = 12000) -> str:
    reader_class = None
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name, fromlist=["PdfReader"])
        except ImportError:
            continue
        reader_class = getattr(module, "PdfReader", None)
        if reader_class is not None:
            break
    if reader_class is None:
        return ""

    chunks = []
    try:
        reader = reader_class(str(path))
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
            if sum(len(chunk) for chunk in chunks) >= max_chars:
                break
    except Exception:
        return ""
    return clean_text(" ".join(chunks)[:max_chars])


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(clean_text(x) for x in value if clean_text(x))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {clean_text(v)}" for k, v in value.items())
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_arm_label(label: str, max_len: int = 240) -> str:
    text = clean_text(label)
    text = re.sub(r"\s*\([^)]{80,}\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def get_nested(obj: dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_count_percent(text: str) -> dict[str, Any]:
    match = re.search(r"(?P<count>\d+(?:\.\d+)?)\s*(?:\((?P<pct>\d+(?:\.\d+)?)%\))?", text)
    if not match:
        return {"raw": text}
    out: dict[str, Any] = {"raw": text, "count": float(match.group("count"))}
    if match.group("pct") is not None:
        out["percent"] = float(match.group("pct"))
    return out


def parse_enrollment_count(text: str) -> int | None:
    match = re.search(r"\d+", text or "")
    return int(match.group(0)) if match else None


def extract_outcomes(results_posted: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = results_posted.get("5. Outcome measures", [])
    extracted: list[dict[str, Any]] = []
    if not isinstance(outcomes, list):
        return extracted

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        table = outcome.get("Data Table", [])
        measurements = []
        denominators = []
        if isinstance(table, list):
            for row in table:
                if not isinstance(row, dict):
                    continue
                category = clean_text(row.get("Category")).lower()
                if category == "measurement":
                    for arm, value in row.items():
                        if arm != "Category":
                            measurements.append(
                                {
                                    "arm": compact_arm_label(arm),
                                    **parse_count_percent(clean_text(value)),
                                }
                            )
                elif category.startswith("denominator"):
                    for arm, value in row.items():
                        if arm != "Category":
                            parsed = parse_count_percent(clean_text(value))
                            denominators.append(
                                {
                                    "arm": compact_arm_label(arm),
                                    "denominator": parsed.get("count"),
                                    "raw": parsed.get("raw", clean_text(value)),
                                }
                            )
        denominator_by_arm = {
            d["arm"]: d.get("denominator")
            for d in denominators
            if d.get("denominator") is not None
        }
        arm_results = []
        for measurement in measurements:
            denominator = denominator_by_arm.get(measurement["arm"])
            result = dict(measurement)
            if denominator is not None:
                result["denominator"] = denominator
                if "count" in result and denominator:
                    result["proportion"] = round(float(result["count"]) / float(denominator), 6)
            arm_results.append(result)
        extracted.append(
            {
                "type": clean_text(outcome.get("Type")),
                "title": clean_text(outcome.get("Title")),
                "description": clean_text(outcome.get("Description")),
                "time_frame": clean_text(outcome.get("Time Frame")),
                "population_description": clean_text(outcome.get("Population Description")),
                "unit": clean_text(outcome.get("Unit of Measure")),
                "param_type": clean_text(outcome.get("Param Type")),
                "denominators": denominators,
                "measurements": measurements,
                "arm_results": arm_results,
            }
        )
    return extracted


def infer_endpoint_family(title: str) -> str:
    low = title.lower()
    rules = [
        ("progression free", "PFS"),
        ("progression-free", "PFS"),
        ("event free", "EFS"),
        ("event-free", "EFS"),
        ("disease free", "DFS"),
        ("disease-free", "DFS"),
        ("relapse free", "RFS"),
        ("relapse-free", "RFS"),
        ("duration of response", "DOR"),
        ("overall survival", "OS"),
        ("disease control", "DCR"),
        ("response", "ORR/CR/PR"),
        ("complete response", "CR"),
        ("partial response", "PR"),
        ("minimal residual", "MRD"),
        ("mrd", "MRD"),
        ("dose limiting", "DLT"),
        ("adverse", "Safety/AE"),
        ("toxicity", "Safety/AE"),
        ("discontinuation", "Treatment discontinuation"),
    ]
    for needle, family in rules:
        if needle in low:
            return family
    if re.search(r"\bsurvival\b", low):
        return "OS"
    return "Other"


def infer_oncology_concepts(text: str) -> dict[str, Any]:
    low = text.lower()

    histology_rules = [
        (r"\bdiffuse large b[- ]cell lymphoma\b|\bdlbcl\b", "DLBCL"),
        (r"\bprimary mediastinal b[- ]cell lymphoma\b|\bpmbl\b", "PMBL"),
        (r"\bburkitt\b", "Burkitt lymphoma"),
        (r"\bmantle cell\b", "Mantle cell lymphoma"),
        (r"\bnon[- ]hodgkin", "Non-Hodgkin lymphoma"),
        (r"(?<!non[- ])(?<!non )\bhodgkin", "Hodgkin lymphoma"),
        (r"\bmultiple myeloma\b", "Multiple myeloma"),
        (r"\bacute lymphoblastic\b", "ALL"),
        (r"\bacute myeloid\b|\baml\b", "AML"),
        (r"\bchronic lymphocytic\b|\bcll\b", "CLL"),
        (r"\bnon[- ]small cell lung\b|\bnsclc\b", "NSCLC"),
        (r"\bsmall cell lung\b|\bsclc\b", "SCLC"),
        (r"\bbreast cancer\b", "Breast cancer"),
        (r"\bovarian\b", "Ovarian cancer"),
        (r"\bcolorectal\b", "Colorectal cancer"),
        (r"\bpancreatic\b", "Pancreatic cancer"),
        (r"\bprostate\b", "Prostate cancer"),
        (r"\bmelanoma\b", "Melanoma"),
        (r"\bglioblastoma\b", "Glioblastoma"),
    ]
    marker_rules = [
        (r"\bcd20\b", "CD20"),
        (r"\bher2\b", "HER2"),
        (r"\begfr\b", "EGFR"),
        (r"\balk\b", "ALK"),
        (r"\bbraf\b", "BRAF"),
        (r"\bkras\b", "KRAS"),
        (r"\bbcl[- ]?2\b", "BCL2"),
        (r"\bpd[- ]?l1\b", "PD-L1"),
        (r"\bmsi\b", "MSI"),
        (r"\bmismatch repair\b", "MMR"),
    ]
    primary_site_rules = [
        (r"\blymphoma\b", "Hematologic malignancy"),
        (r"\bleukemia\b", "Hematologic malignancy"),
        (r"\bmyeloma\b", "Hematologic malignancy"),
        (r"\blung\b", "Lung"),
        (r"\bbreast\b", "Breast"),
        (r"\bovarian\b", "Ovary"),
        (r"\bcolorectal\b|\bcolon\b|\brectal\b", "Colorectal"),
        (r"\bpancreatic\b", "Pancreas"),
        (r"\bprostate\b", "Prostate"),
        (r"\bmelanoma\b", "Skin"),
        (r"\bglioblastoma\b", "Brain/CNS"),
    ]

    histologies = [label for pattern, label in histology_rules if re.search(pattern, low)]
    markers = [label for pattern, label in marker_rules if re.search(pattern, low)]
    primary_sites = [label for pattern, label in primary_site_rules if re.search(pattern, low)]

    if re.search(r"relapsed|refractory|\br/r\b|previously treated", low):
        line = "Relapsed/refractory or previously treated"
    elif re.search(r"\bpreviously untreated\b|\buntreated\b|\bfront[- ]?line\b|\bfirst[- ]?line\b|newly diagnosed", low):
        line = "Frontline / previously untreated"
    elif "maintenance" in low:
        line = "Maintenance"
    elif "neoadjuvant" in low or "neo-adjuvant" in low:
        line = "Neoadjuvant"
    elif "adjuvant" in low:
        line = "Adjuvant"
    else:
        line = "Not reported"

    stage_matches = sorted(set(re.findall(r"\bstage\s+(?:i{1,3}|iv|[1-4][a-d]?)\b", low)))
    age = "Pediatric and adult" if re.search(r"children|pediatric|paediatric", low) and re.search(r"adult", low) else (
        "Pediatric" if re.search(r"children|pediatric|paediatric", low) else (
            "Adult" if re.search(r"\badult", low) else "Not reported"
        )
    )

    return {
        "primary_site": sorted(set(primary_sites)) or ["Not normalized"],
        "histology": sorted(set(histologies)) or ["Not normalized"],
        "molecular_marker": sorted(set(markers)) or ["Not reported"],
        "stage_or_risk": stage_matches or ["Not reported"],
        "line_of_therapy": line,
        "age": age,
    }


def infer_intervention_concepts(interventions: Any, text: str) -> dict[str, Any]:
    combined = f"{clean_text(interventions)} {text}".lower()
    drug_class_rules = [
        (r"\brituximab\b", "Anti-CD20 antibody"),
        (r"\bofatumumab\b", "Anti-CD20 antibody"),
        (r"\bcheckpoint\b", "Immunotherapy/checkpoint inhibitor"),
        (r"\bpembrolizumab\b", "PD-1/PD-L1 inhibitor"),
        (r"\bnivolumab\b", "PD-1/PD-L1 inhibitor"),
        (r"\batezolizumab\b", "PD-1/PD-L1 inhibitor"),
        (r"\bdurvalumab\b", "PD-1/PD-L1 inhibitor"),
        (r"\bchemotherapy\b", "Chemotherapy"),
        (r"\betoposide\b", "Chemotherapy"),
        (r"\bcyclophosphamide\b", "Chemotherapy"),
        (r"\bdoxorubicin\b", "Chemotherapy"),
        (r"\bvincristine\b", "Chemotherapy"),
        (r"\bprednisone\b", "Corticosteroid"),
        (r"\bibrutinib\b", "BTK inhibitor"),
        (r"\bacalabrutinib\b", "BTK inhibitor"),
        (r"\bbortezomib\b", "Proteasome inhibitor"),
        (r"\bcarfilzomib\b", "Proteasome inhibitor"),
        (r"\blenalidomide\b", "Immunomodulatory drug"),
        (r"\bcopanlisib\b", "PI3K inhibitor"),
        (r"\bbevacizumab\b", "Anti-VEGF antibody"),
    ]
    backbone_rules = [
        (r"\bda[- ]?epoch[- ]?r\b|\bdose[- ]adjusted epoch[- ]rituximab\b", "DA-EPOCH-R"),
        (r"\bepoch[- ]?r\b|\br[- ]?epoch\b", "EPOCH-R"),
        (r"\bepoch\b", "EPOCH"),
        (r"\br[- ]?chop\b", "R-CHOP"),
        (r"\bchop\b", "CHOP"),
        (r"\br[- ]?ice\b", "R-ICE"),
        (r"\bifosfamide\b.*\bcarboplatin\b.*\betoposide\b|\bice regimen\b", "ICE"),
        (r"\bgemcitabine\b", "Gemcitabine-based"),
        (r"\bcisplatin\b|\bcarboplatin\b", "Platinum-based"),
    ]
    classes = [label for pattern, label in drug_class_rules if re.search(pattern, combined)]
    backbones = [label for pattern, label in backbone_rules if re.search(pattern, combined)]
    if "DA-EPOCH-R" in backbones:
        backbones = [b for b in backbones if b not in {"EPOCH-R", "EPOCH"}]
    elif "EPOCH-R" in backbones:
        backbones = [b for b in backbones if b != "EPOCH"]
    return {
        "drug_classes": sorted(set(classes)),
        "backbone_regimen": sorted(set(backbones)) or ["Not normalized"],
    }


def infer_design_concepts(design: dict[str, Any], outcomes: list[dict[str, Any]]) -> dict[str, str]:
    allocation = clean_text(design.get("Allocation")).lower()
    model = clean_text(design.get("Interventional Model")).lower()
    if "non_random" in allocation or "non-random" in allocation or "non random" in allocation:
        randomized = "No"
    elif "random" in allocation:
        randomized = "Yes"
    else:
        randomized = "Not reported" if not allocation else "No"
    if "single" in model:
        arm_structure = "Single-arm"
    elif "parallel" in model or "factorial" in model or "crossover" in model:
        arm_structure = "Multi-arm"
    else:
        arm_structure = "Not reported"
    number_of_arms = "Not reported"
    if outcomes:
        arms = set()
        for outcome in outcomes:
            for row in outcome.get("arm_results", []):
                arms.add(row.get("arm", ""))
        if arms:
            number_of_arms = str(len(arms))
    return {
        "single_or_multi_arm": arm_structure,
        "randomized": randomized,
        "number_of_arms": number_of_arms,
    }


def summarize_borrowable_quantities(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quantities = []
    for outcome in primary:
        usable_results = []
        for row in outcome.get("arm_results", []):
            usable_results.append(
                {
                    "arm": row.get("arm", ""),
                    "count": row.get("count"),
                    "denominator": row.get("denominator"),
                    "percent": row.get("percent"),
                    "proportion": row.get("proportion"),
                    "raw": row.get("raw", ""),
                }
            )
        quantities.append(
            {
                "endpoint": outcome.get("title", ""),
                "endpoint_family": outcome.get("endpoint_family", ""),
                "unit": outcome.get("unit", ""),
                "param_type": outcome.get("param_type", ""),
                "time_frame": outcome.get("time_frame", ""),
                "arm_results": usable_results,
            }
        )
    return quantities


def extract_trial_record(folder: Path) -> TrialRecord | None:
    json_path = find_trial_json(folder)
    if json_path is None:
        return None
    pdfs = find_supporting_pdfs(folder)
    raw = read_json(json_path)
    details = raw.get("Study details", {})
    results = raw.get("Results Posted", {})
    overview = get_nested(details, "5. Study Overview", default={})
    design = get_nested(results, "2. Study Design", default={})
    dates = get_nested(results, "4. Study Record Dates", default={})
    outcomes = extract_outcomes(results if isinstance(results, dict) else {})

    nct_id = clean_text(details.get("1. NCT number")) or folder.name
    interventions = results.get("1. Intervention/Treatment", []) if isinstance(results, dict) else []
    protocol_excerpt = read_pdf_excerpt(pdfs["protocol"], max_chars=8000)
    sap_excerpt = read_pdf_excerpt(pdfs["sap"], max_chars=5000)
    extracted = {
        "nct_id": nct_id,
        "json_path": str(json_path),
        "brief_title": clean_text(overview.get("Brief Title") if isinstance(overview, dict) else ""),
        "official_title": clean_text(overview.get("Official Title") if isinstance(overview, dict) else ""),
        "brief_summary": clean_text(overview.get("Brief Summary") if isinstance(overview, dict) else ""),
        "detailed_description": clean_text(overview.get("Detailed Description") if isinstance(overview, dict) else ""),
        "status": clean_text(details.get("2. Study status") if isinstance(details, dict) else ""),
        "phase": clean_text(details.get("7. Phase") if isinstance(details, dict) else ""),
        "interventions": interventions,
        "design": design if isinstance(design, dict) else {},
        "enrollment": clean_text(results.get("3. Enrollment (Actual)") if isinstance(results, dict) else ""),
        "dates": dates if isinstance(dates, dict) else {},
        "supporting_documents": {
            "protocol_pdf": str(pdfs["protocol"]) if pdfs["protocol"] else "",
            "sap_pdf": str(pdfs["sap"]) if pdfs["sap"] else "",
            "protocol_excerpt": protocol_excerpt,
            "sap_excerpt": sap_excerpt,
        },
        "outcomes": [
            {**o, "endpoint_family": infer_endpoint_family(o["title"])}
            for o in outcomes
        ],
    }
    return TrialRecord(
        nct_id=nct_id,
        folder=folder,
        json_path=json_path,
        protocol_path=pdfs["protocol"],
        sap_path=pdfs["sap"],
        raw_json=raw,
        extracted=extracted,
    )


def make_rule_based_summary(extracted: dict[str, Any]) -> dict[str, Any]:
    primary = [o for o in extracted["outcomes"] if o.get("type") == "PRIMARY"]
    secondary = [o for o in extracted["outcomes"] if o.get("type") != "PRIMARY"]
    source_text = " ".join(
        [
            extracted.get("brief_title", ""),
            extracted.get("official_title", ""),
            extracted.get("brief_summary", ""),
            extracted.get("detailed_description", ""),
            clean_text(extracted.get("interventions", "")),
            extracted.get("supporting_documents", {}).get("protocol_excerpt", ""),
            extracted.get("supporting_documents", {}).get("sap_excerpt", ""),
        ]
    )
    oncology = infer_oncology_concepts(source_text)
    intervention = infer_intervention_concepts(extracted.get("interventions", ""), source_text)
    design_concepts = infer_design_concepts(extracted.get("design", {}), extracted.get("outcomes", []))
    enrollment_count = parse_enrollment_count(extracted.get("enrollment", ""))
    denominators = [
        {
            "endpoint": outcome.get("title", ""),
            "endpoint_family": outcome.get("endpoint_family", ""),
            "denominators": outcome.get("denominators", []),
        }
        for outcome in extracted["outcomes"]
        if outcome.get("denominators")
    ]
    borrowable_quantities = summarize_borrowable_quantities(primary)
    nonborrowability_risks = []
    if not extracted.get("outcomes"):
        nonborrowability_risks.append("No posted outcome results available in JSON.")
    if oncology["line_of_therapy"] == "Not reported":
        nonborrowability_risks.append("Line of therapy was not normalized from available text.")
    if oncology["histology"] == ["Not normalized"]:
        nonborrowability_risks.append("Cancer histology was not normalized from available text.")

    full_text = " ".join(
        [
            extracted.get("brief_title", ""),
            extracted.get("brief_summary", ""),
            extracted.get("detailed_description", ""),
            clean_text(extracted.get("interventions", "")),
            clean_text(oncology),
            clean_text(intervention),
            clean_text(primary[:3]),
        ]
    )
    return {
        "nct_id": extracted["nct_id"],
        "brief_title": extracted.get("brief_title", ""),
        "phase": extracted.get("phase", ""),
        "status": extracted.get("status", ""),
        "cancer_type": {
            "primary_site": oncology["primary_site"],
            "histology": oncology["histology"],
            "molecular_marker": oncology["molecular_marker"],
            "stage_or_risk": oncology["stage_or_risk"],
            "line_of_therapy": oncology["line_of_therapy"],
            "prior_treatment": "Not reported",
        },
        "population": {
            "age": oncology["age"],
            "key_inclusion": [],
            "key_exclusion": [],
            "performance_status": "Not reported",
            "subgroups": [],
        },
        "intervention": {
            "experimental_regimen": clean_text(extracted.get("interventions", "")),
            "control_or_comparator": "Not reported",
            "drug_classes": intervention["drug_classes"],
            "backbone_regimen": intervention["backbone_regimen"],
            "dose_schedule_summary": "See source record",
            "treatment_duration": "Not reported",
        },
        "design": {
            "allocation": clean_text(extracted.get("design", {}).get("Allocation")),
            "interventional_model": clean_text(extracted.get("design", {}).get("Interventional Model")),
            "masking": clean_text(extracted.get("design", {}).get("Masking")),
            "primary_purpose": clean_text(extracted.get("design", {}).get("Primary Purpose")),
            "single_or_multi_arm": design_concepts["single_or_multi_arm"],
            "randomized": design_concepts["randomized"],
            "sample_size": extracted.get("enrollment", ""),
            "sample_size_n": enrollment_count,
            "number_of_arms": design_concepts["number_of_arms"],
            "follow_up": "Not reported",
        },
        "endpoints": {
            "primary": primary,
            "secondary_or_other": secondary[:10],
        },
        "results": {
            "has_posted_results": bool(extracted.get("outcomes")),
            "primary_results": primary,
            "safety_results": [
                o for o in extracted["outcomes"] if "Safety" in o.get("endpoint_family", "")
            ],
            "denominators": denominators,
            "follow_up_duration": "Not normalized",
        },
        "borrowing_relevance": {
            "borrowable_quantities": borrowable_quantities,
            "major_similarity_drivers": [
                f"Histology: {', '.join(oncology['histology'])}",
                f"Line of therapy: {oncology['line_of_therapy']}",
                f"Backbone regimen: {', '.join(intervention['backbone_regimen'])}",
                f"Primary endpoint families: {', '.join(sorted(set(o.get('endpoint_family', '') for o in primary if o.get('endpoint_family')))) or 'Not reported'}",
            ],
            "major_nonborrowability_risks": nonborrowability_risks,
            "notes": "Rule-based fallback summary. Use LLM prompt for stronger normalization.",
        },
        "source_documents": {
            "json_path": extracted.get("json_path", ""),
            "protocol_pdf": extracted.get("supporting_documents", {}).get("protocol_pdf", ""),
            "sap_pdf": extracted.get("supporting_documents", {}).get("sap_pdf", ""),
            "protocol_text_available": bool(extracted.get("supporting_documents", {}).get("protocol_excerpt")),
            "sap_text_available": bool(extracted.get("supporting_documents", {}).get("sap_excerpt")),
        },
        "one_paragraph_summary_for_embedding": full_text[:4000],
    }


def aspect_text(summary: dict[str, Any], aspect: str) -> str:
    if aspect == "disease_population":
        return clean_text({"cancer_type": summary["cancer_type"], "population": summary["population"]})
    if aspect == "intervention":
        return clean_text(summary["intervention"])
    if aspect == "endpoint":
        return clean_text(summary["endpoints"])
    if aspect == "design":
        return clean_text(summary["design"])
    if aspect == "results_safety":
        return clean_text(summary["results"])
    return clean_text(summary)


TRIAL2VEC_COLUMNS = (
    "nct_id",
    "description",
    "title",
    "intervention_name",
    "disease",
    "keyword",
    "outcome_measure",
    "criteria",
    "reference",
    "overall_status",
)


def summary_to_trial2vec_row(summary: dict[str, Any]) -> dict[str, str]:
    summary = summary if isinstance(summary, dict) else {}
    endpoints = summary.get("endpoints") or {}
    endpoints = endpoints if isinstance(endpoints, dict) else {}
    primary_endpoints = endpoints.get("primary") or []
    if isinstance(primary_endpoints, dict):
        primary_endpoints = [primary_endpoints]
    if not isinstance(primary_endpoints, (list, tuple)):
        primary_endpoints = []
    endpoint_titles = [
        clean_text(endpoint.get("title", ""))
        for endpoint in primary_endpoints
        if isinstance(endpoint, dict) and clean_text(endpoint.get("title", ""))
    ]
    population = summary.get("population") or {}
    population = population if isinstance(population, dict) else {}
    inclusion = clean_text(population.get("key_inclusion", []))
    exclusion = clean_text(population.get("key_exclusion", []))
    criteria = clean_text(
        {
            label: value
            for label, value in (("inclusion", inclusion), ("exclusion", exclusion))
            if value
        }
    )
    intervention = summary.get("intervention") or {}
    intervention = intervention if isinstance(intervention, dict) else {}
    cancer_type = summary.get("cancer_type") or {}
    cancer_type = cancer_type if isinstance(cancer_type, dict) else {}
    design = summary.get("design") or {}
    design = design if isinstance(design, dict) else {}
    row = {
        "nct_id": clean_text(summary.get("nct_id", "")),
        "description": clean_text(
            [
                summary.get("brief_summary", ""),
                summary.get("one_paragraph_summary_for_embedding", ""),
            ]
        ),
        "title": clean_text(summary.get("brief_title", summary.get("title", ""))),
        "intervention_name": clean_text(intervention),
        "disease": clean_text(cancer_type),
        "keyword": clean_text(
            [
                summary.get("phase", ""),
                summary.get("status", ""),
                design,
            ]
        ),
        "outcome_measure": clean_text(endpoint_titles),
        "criteria": criteria,
        "reference": "",
        "overall_status": clean_text(summary.get("status", "")),
    }
    return {column: row[column] for column in TRIAL2VEC_COLUMNS}


def hashing_embedding(text: str, dim: int = 2048) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+\-/_.]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class TextEmbedder:
    backend_name = "base"
    model_name = ""

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(TextEmbedder):
    backend_name = "hashing"
    model_name = "signed-token-hashing-2048"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([hashing_embedding(text) for text in texts])


class ClinicalBertEmbedder(TextEmbedder):
    backend_name = "clinicalbert"

    def __init__(
        self,
        model_name: str = DEFAULT_CLINICALBERT_MODEL,
        batch_size: int = 16,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ClinicalBERT embedding requires torch and transformers in the active Python environment."
            ) from exc

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        if device is None:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            use_safetensors=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        torch = self.torch
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(vectors)


def make_embedder(
    backend: str,
    model_name: str = DEFAULT_CLINICALBERT_MODEL,
    batch_size: int = 16,
    max_length: int = 256,
) -> TextEmbedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "clinicalbert":
        return ClinicalBertEmbedder(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")


def summary_embedding(summary: dict[str, Any], embedder: TextEmbedder | None = None) -> dict[str, np.ndarray]:
    embedder = embedder or HashingEmbedder()
    return {
        aspect: embedder.encode([aspect_text(summary, aspect)])[0]
        for aspect in ASPECT_WEIGHTS
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def weighted_similarity(query: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    by_aspect = {
        aspect: cosine(query[aspect], candidate[aspect])
        for aspect in ASPECT_WEIGHTS
    }
    score = sum(ASPECT_WEIGHTS[a] * by_aspect[a] for a in ASPECT_WEIGHTS)
    return score, by_aspect


def score_trial2vec_index(
    query_vector: np.ndarray,
    trial2vec_index_path: Path,
    excluded_nct_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    index = np.load(trial2vec_index_path, allow_pickle=False)
    nct_ids = index["nct_ids"]
    embeddings = index["embeddings"]
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    scored = []
    for idx, nct_id_raw in enumerate(nct_ids):
        nct_id = str(nct_id_raw)
        if nct_id == excluded_nct_id:
            continue
        score = cosine(query, np.asarray(embeddings[idx], dtype=np.float32).reshape(-1))
        scored.append(
            {
                "nct_id": nct_id,
                "score": score,
                "score_0_100": round(100 * max(0.0, score), 2),
                "aspect_scores": {},
                "retrieval_backend": "trial2vec",
            }
        )
    scored.sort(key=lambda row: row["score"], reverse=True)
    top_rows = scored[:top_k]
    for rank, row in enumerate(top_rows, start=1):
        row["retrieval_rank"] = rank
    return top_rows


def make_compatible_torch_load(original_torch_load: Any) -> Any:
    try:
        supports_weights_only = "weights_only" in inspect.signature(original_torch_load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        if supports_weights_only and "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    return compatible_torch_load


def make_compatible_load_state_dict(original_load_state_dict: Any) -> Any:
    def compatible_load_state_dict(
        module: Any,
        state_dict: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not args and "strict" not in kwargs:
            kwargs["strict"] = False
        return original_load_state_dict(module, state_dict, *args, **kwargs)

    return compatible_load_state_dict


def ensure_pandas_applymap_compat(pd: Any) -> None:
    if not hasattr(pd.DataFrame, "applymap") and hasattr(pd.DataFrame, "map"):
        pd.DataFrame.applymap = pd.DataFrame.map


def load_trial2vec_search_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pandas as pd
        import torch
        from trial2vec import Trial2Vec
    except ImportError as exc:
        raise RuntimeError(
            "Trial2Vec retrieval requires optional Trial2Vec search dependencies: "
            "pandas, torch, and trial2vec."
        ) from exc
    return pd, torch, Trial2Vec


def encode_trial2vec_query(query_row: dict[str, str], trial2vec_model_dir: Path) -> np.ndarray:
    pd, torch, Trial2Vec = load_trial2vec_search_dependencies()
    ensure_pandas_applymap_compat(pd)
    model = Trial2Vec(device="cpu")

    original_torch_load = torch.load
    original_load_state_dict = torch.nn.Module.load_state_dict
    torch.load = make_compatible_torch_load(original_torch_load)
    torch.nn.Module.load_state_dict = make_compatible_load_state_dict(original_load_state_dict)
    try:
        model.from_pretrained(str(trial2vec_model_dir))
    finally:
        torch.load = original_torch_load
        torch.nn.Module.load_state_dict = original_load_state_dict

    _, embeddings = model.encode({"x": pd.DataFrame([query_row])}, return_dict=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] < 1:
        raise RuntimeError("Trial2Vec did not return a query embedding.")
    return embeddings[0]


def build_index(
    db_root: Path,
    output_dir: Path,
    embedding_backend: str = "hashing",
    embedding_model: str = DEFAULT_CLINICALBERT_MODEL,
    embedding_batch_size: int = 16,
    embedding_max_length: int = 256,
) -> None:
    if not db_root.exists():
        raise FileNotFoundError(f"Database root does not exist: {db_root}")
    if not db_root.is_dir():
        raise NotADirectoryError(f"Database root is not a directory: {db_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = output_dir / "trial_summaries.jsonl"
    embeddings_path = output_dir / "trial_embeddings.npz"

    folders = sorted([p for p in db_root.iterdir() if p.is_dir() and p.name.startswith("NCT")])
    summaries: list[dict[str, Any]] = []
    nct_ids: list[str] = []

    with summaries_path.open("w", encoding="utf-8") as out:
        for i, folder in enumerate(folders, start=1):
            record = extract_trial_record(folder)
            if record is None:
                continue
            summary = make_rule_based_summary(record.extracted)
            summaries.append(summary)
            nct_ids.append(summary["nct_id"])
            out.write(json.dumps(summary, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(f"Indexed {i}/{len(folders)} folders")

    if not nct_ids:
        raise RuntimeError(
            f"No trials were indexed. Check that {db_root} contains NCT* folders with JSON files."
        )

    print(f"Encoding {len(summaries)} trials with {embedding_backend} embeddings...")
    embedder = make_embedder(
        embedding_backend,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        max_length=embedding_max_length,
    )
    embeddings = {}
    for aspect in ASPECT_WEIGHTS:
        texts = [aspect_text(summary, aspect) for summary in summaries]
        embeddings[aspect] = embedder.encode(texts)
        print(f"Encoded aspect: {aspect} ({embeddings[aspect].shape[1]} dims)")

    arrays = {
        "nct_ids": np.array(nct_ids),
        "embedding_backend": np.array([embedder.backend_name]),
        "embedding_model": np.array([embedder.model_name]),
        **embeddings,
    }
    np.savez_compressed(embeddings_path, **arrays)
    print(f"Wrote {summaries_path}")
    print(f"Wrote {embeddings_path}")


def load_summaries(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records[item["nct_id"]] = {
                    "nct_id": item.get("nct_id", ""),
                    "brief_title": item.get("brief_title", ""),
                    "phase": item.get("phase", ""),
                    "status": item.get("status", ""),
                    "cancer_type": item.get("cancer_type", {}),
                    "population": item.get("population", {}),
                    "intervention": item.get("intervention", {}),
                    "design": item.get("design", {}),
                    "endpoints": item.get("endpoints", {}),
                    "results": {
                        "has_posted_results": item.get("results", {}).get("has_posted_results"),
                        "denominators": item.get("results", {}).get("denominators", []),
                        "follow_up_duration": item.get("results", {}).get("follow_up_duration", ""),
                    },
                    "borrowing_relevance": item.get("borrowing_relevance", {}),
                    "source_documents": item.get("source_documents", {}),
                }
    return records


def enrich_candidate_row(row: dict[str, Any], candidate_summary: dict[str, Any]) -> dict[str, Any]:
    borrowing = candidate_summary.get("borrowing_relevance", {})
    row.update(
        {
            "title": candidate_summary.get("brief_title", ""),
            "phase": candidate_summary.get("phase", ""),
            "status": candidate_summary.get("status", ""),
            "cancer_type": candidate_summary.get("cancer_type", {}),
            "population": candidate_summary.get("population", {}),
            "intervention": candidate_summary.get("intervention", {}),
            "design": candidate_summary.get("design", {}),
            "endpoints": candidate_summary.get("endpoints", {}),
            "results": candidate_summary.get("results", {}),
            "result_usability": {
                "has_posted_results": candidate_summary.get("results", {}).get("has_posted_results"),
                "denominators_available": bool(candidate_summary.get("results", {}).get("denominators")),
                "source_documents": candidate_summary.get("source_documents", {}),
            },
            "similarity_drivers": borrowing.get("major_similarity_drivers", []),
            "nonborrowability_risks": borrowing.get("major_nonborrowability_risks", []),
            "borrowable_quantities": borrowing.get("borrowable_quantities", []),
        }
    )
    return row


def normalized_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(x) for x in value]
    else:
        values = [str(value)]
    ignored = {"", "not reported", "not normalized", "none", "unknown"}
    return {v.strip().lower() for v in values if v and v.strip().lower() not in ignored}


def overlap_score(query_values: Any, candidate_values: Any) -> tuple[float, list[str]]:
    q = normalized_values(query_values)
    c = normalized_values(candidate_values)
    if not q or not c:
        return 0.0, []
    overlap = sorted(q & c)
    denom = max(len(q), len(c), 1)
    return len(overlap) / denom, overlap


def primary_endpoint_families(summary: dict[str, Any]) -> list[str]:
    endpoints = summary.get("endpoints", {}).get("primary", [])
    families = []
    for endpoint in endpoints:
        family = endpoint.get("endpoint_family")
        if family:
            families.append(family)
    return sorted(set(families))


def candidate_endpoint_families(candidate: dict[str, Any]) -> list[str]:
    families = []
    for quantity in candidate.get("borrowable_quantities", []):
        family = quantity.get("endpoint_family")
        if family:
            families.append(family)
    if not families:
        endpoints = candidate.get("endpoints", {}).get("primary", [])
        for endpoint in endpoints:
            family = endpoint.get("endpoint_family")
            if family:
                families.append(family)
    return sorted(set(families))


def has_matched_endpoint_arm_results(candidate: dict[str, Any], query_families: list[str]) -> bool:
    query_family_set = set(query_families)
    for quantity in candidate.get("borrowable_quantities", []):
        family = quantity.get("endpoint_family")
        if family not in query_family_set:
            continue
        for row in quantity.get("arm_results", []):
            if row.get("count") is not None and row.get("denominator") is not None:
                return True
    return False


def has_usable_arm_results(candidate: dict[str, Any]) -> bool:
    for quantity in candidate.get("borrowable_quantities", []):
        for row in quantity.get("arm_results", []):
            if row.get("count") is not None and row.get("denominator") is not None:
                return True
    return False


def score_phase_match(query_phase: str, candidate_phase: str) -> float:
    q = clean_text(query_phase).lower()
    c = clean_text(candidate_phase).lower()
    if not q or not c:
        return 0.2
    if q == c:
        return 1.0
    q_num = re.search(r"\d+", q)
    c_num = re.search(r"\d+", c)
    if q_num and c_num and abs(int(q_num.group()) - int(c_num.group())) <= 1:
        return 0.65
    return 0.25


def score_prior_borrowing_pair(
    query_summary: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    red_flags: list[str] = list(candidate.get("nonborrowability_risks", []))
    explanations: list[str] = []

    q_cancer = query_summary.get("cancer_type", {})
    c_cancer = candidate.get("cancer_type", {})
    hist_overlap, hist_terms = overlap_score(q_cancer.get("histology"), c_cancer.get("histology"))
    site_overlap, site_terms = overlap_score(q_cancer.get("primary_site"), c_cancer.get("primary_site"))
    marker_overlap, marker_terms = overlap_score(q_cancer.get("molecular_marker"), c_cancer.get("molecular_marker"))
    q_line = clean_text(q_cancer.get("line_of_therapy")).lower()
    c_line = clean_text(c_cancer.get("line_of_therapy")).lower()
    line_match = 1.0 if q_line and q_line == c_line and q_line != "not reported" else 0.0
    disease_raw = 0.45 * hist_overlap + 0.25 * site_overlap + 0.15 * marker_overlap + 0.15 * line_match
    disease_score = round(5 * disease_raw, 2)
    if hist_terms:
        explanations.append(f"Shared histology: {', '.join(hist_terms)}.")
    if site_terms:
        explanations.append(f"Shared primary site/category: {', '.join(site_terms)}.")
    if marker_terms:
        explanations.append(f"Shared marker: {', '.join(marker_terms)}.")
    if q_line != c_line and q_line != "not reported" and c_line != "not reported":
        red_flags.append(f"Treatment line mismatch: query={q_cancer.get('line_of_therapy')} candidate={c_cancer.get('line_of_therapy')}.")

    q_intervention = query_summary.get("intervention", {})
    c_intervention = candidate.get("intervention", {})
    backbone_overlap, backbone_terms = overlap_score(
        q_intervention.get("backbone_regimen"),
        c_intervention.get("backbone_regimen"),
    )
    class_overlap, class_terms = overlap_score(
        q_intervention.get("drug_classes"),
        c_intervention.get("drug_classes"),
    )
    treatment_raw = 0.7 * backbone_overlap + 0.3 * class_overlap
    treatment_score = round(5 * treatment_raw, 2)
    if backbone_terms:
        explanations.append(f"Shared regimen backbone: {', '.join(backbone_terms)}.")
    elif normalized_values(q_intervention.get("backbone_regimen")):
        red_flags.append("No normalized regimen-backbone overlap.")
    if class_terms:
        explanations.append(f"Shared drug classes: {', '.join(class_terms)}.")

    q_endpoints = primary_endpoint_families(query_summary)
    c_endpoints = candidate_endpoint_families(candidate)
    endpoint_overlap, endpoint_terms = overlap_score(q_endpoints, c_endpoints)
    has_matched_denominators = has_matched_endpoint_arm_results(candidate, q_endpoints)
    endpoint_raw = 0.75 * endpoint_overlap + 0.25 * (1.0 if has_matched_denominators else 0.0)
    endpoint_score = round(5 * endpoint_raw, 2)
    if endpoint_terms:
        explanations.append(f"Shared primary endpoint families: {', '.join(endpoint_terms)}.")
    else:
        red_flags.append("No primary endpoint-family overlap.")

    q_design = query_summary.get("design", {})
    c_design = candidate.get("design", {})
    same_arm_structure = (
        clean_text(q_design.get("single_or_multi_arm")).lower()
        == clean_text(c_design.get("single_or_multi_arm")).lower()
        and clean_text(q_design.get("single_or_multi_arm")).lower() not in {"", "not reported"}
    )
    same_randomization = (
        clean_text(q_design.get("randomized")).lower()
        == clean_text(c_design.get("randomized")).lower()
        and clean_text(q_design.get("randomized")).lower() not in {"", "not reported"}
    )
    phase_match = score_phase_match(query_summary.get("phase", ""), candidate.get("phase", ""))
    design_raw = 0.4 * phase_match + 0.3 * float(same_arm_structure) + 0.3 * float(same_randomization)
    design_score = round(5 * design_raw, 2)
    if same_arm_structure:
        explanations.append(f"Same arm structure: {q_design.get('single_or_multi_arm')}.")
    if same_randomization:
        explanations.append(f"Same randomization status: {q_design.get('randomized')}.")

    has_results = bool(candidate.get("result_usability", {}).get("has_posted_results"))
    usable_arm_results = has_usable_arm_results(candidate)
    result_raw = 0.4 * float(has_results) + 0.4 * float(usable_arm_results) + 0.2 * float(has_matched_denominators)
    result_score = round(5 * result_raw, 2)
    if not has_results:
        red_flags.append("Candidate has no posted results in indexed JSON.")
    if not usable_arm_results:
        red_flags.append("No arm-level count/denominator pair found for primary borrowable quantities.")

    has_safety = any(
        q.get("endpoint_family") in {"Safety/AE", "DLT", "Treatment discontinuation"}
        for q in candidate.get("borrowable_quantities", [])
    )
    followup_known = clean_text(candidate.get("results", {}).get("follow_up_duration", "")).lower() not in {
        "",
        "not normalized",
        "not reported",
    }
    safety_raw = 0.6 * float(has_safety) + 0.4 * float(followup_known)
    safety_score = round(5 * safety_raw, 2)

    dimension_scores = {
        "disease_population_match": disease_score,
        "treatment_regimen_match": treatment_score,
        "endpoint_estimand_match": endpoint_score,
        "design_phase_match": design_score,
        "result_usability": result_score,
        "safety_and_followup_relevance": safety_score,
    }
    weighted_dimension_score = (
        0.30 * disease_score
        + 0.25 * treatment_score
        + 0.20 * endpoint_score
        + 0.10 * design_score
        + 0.10 * result_score
        + 0.05 * safety_score
    )
    clinical_score = 20 * weighted_dimension_score
    retrieval_score = float(candidate.get("score_0_100", 0.0))
    overall = round(0.75 * clinical_score + 0.25 * retrieval_score, 2)

    if disease_score < 1.5:
        red_flags.append("Low disease/population match.")
    if treatment_score < 1.5:
        red_flags.append("Low treatment-regimen match.")
    if endpoint_score < 1.5:
        red_flags.append("Low endpoint/estimand match.")

    if overall >= 80 and not any("Low " in flag for flag in red_flags):
        suitability = "high"
        discount = 0.75
        adjustments = ["commensurate prior", "sensitivity analysis with robust mixture prior"]
    elif overall >= 60:
        suitability = "medium"
        discount = 0.40
        adjustments = ["robust mixture prior", "discount for residual heterogeneity"]
    elif overall >= 40:
        suitability = "low"
        discount = 0.15
        adjustments = ["sensitivity analysis only", "strong discounting"]
    else:
        suitability = "do_not_borrow"
        discount = 0.0
        adjustments = ["do not borrow in primary analysis"]

    return {
        "candidate_nct_id": candidate.get("nct_id", ""),
        "title": candidate.get("title", ""),
        "retrieval_score": retrieval_score,
        "overall_similarity_score": overall,
        "prior_borrowing_suitability": suitability,
        "suggested_borrowing_discount": discount,
        "dimension_scores": dimension_scores,
        "borrowable_quantities": candidate.get("borrowable_quantities", []),
        "required_adjustments": adjustments,
        "explanation": " ".join(explanations) if explanations else "No strong structured similarity drivers were detected.",
        "red_flags": sorted(set(red_flags)),
        "candidate_snapshot": {
            "phase": candidate.get("phase", ""),
            "status": candidate.get("status", ""),
            "cancer_type": candidate.get("cancer_type", {}),
            "intervention": candidate.get("intervention", {}),
            "design": candidate.get("design", {}),
            "result_usability": candidate.get("result_usability", {}),
        },
    }


def rerank_candidates(
    query_summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    rerank_top_n: int,
) -> list[dict[str, Any]]:
    reranked = [
        score_prior_borrowing_pair(query_summary, candidate)
        for candidate in candidates[:rerank_top_n]
    ]
    reranked.sort(key=lambda x: x["overall_similarity_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
    return reranked


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def canonical_bayesian_endpoint(family: str, title: str = "", time_frame: str = "") -> str | None:
    text = clean_text([family, title, time_frame]).lower()
    if "dor" in text or "duration of response" in text:
        return None
    if "pfs" in text or "progression free" in text or "progression-free" in text:
        if re.search(r"\b6\b|\bsix\b", text) and re.search(r"month|mo\b|months", text):
            return "PFS6"
        return None
    if "orr" in text or "objective response" in text or "response rate" in text:
        return "ORR"
    return None


def arm_role(arm: str) -> str:
    low = clean_text(arm).lower()
    if re.search(r"placebo|control|comparator|standard of care|best supportive|observation", low):
        return "control"
    if re.search(r"experimental|treatment arm|investigational|intervention", low):
        return "treatment"
    return "treatment"


def endpoint_observation_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    count = to_float(row.get("count"))
    denominator = to_float(row.get("denominator"))
    if count is None or denominator is None or denominator <= 0 or count < 0 or count > denominator:
        return None
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator,
    }


def select_arm_observation(
    arm_results: list[dict[str, Any]],
    desired_role: str = "treatment",
) -> tuple[dict[str, Any], dict[str, float]] | None:
    usable = []
    for row in arm_results:
        observation = endpoint_observation_from_row(row)
        if observation is not None:
            usable.append((row, observation))
    if not usable:
        return None
    for row, observation in usable:
        if arm_role(row.get("arm", "")) == desired_role:
            return row, observation
    if desired_role == "treatment":
        return usable[0]
    return None


def query_endpoint_observations(query_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for endpoint in query_summary.get("endpoints", {}).get("primary", []):
        endpoint_key = canonical_bayesian_endpoint(
            endpoint.get("endpoint_family", ""),
            endpoint.get("title", ""),
            endpoint.get("time_frame", ""),
        )
        if endpoint_key is None or endpoint_key in observations:
            continue
        treatment = select_arm_observation(endpoint.get("arm_results", []), "treatment")
        item = {
            "endpoint": endpoint.get("title", ""),
            "endpoint_family": endpoint_key,
            "time_frame": endpoint.get("time_frame", ""),
        }
        if treatment is not None:
            treatment_row, treatment_obs = treatment
            item.update(
                {
                    "treatment_arm": treatment_row.get("arm", ""),
                    "treatment_count": treatment_obs["count"],
                    "treatment_denominator": treatment_obs["denominator"],
                    "treatment_rate": round(treatment_obs["rate"], 6),
                }
            )
            control = select_arm_observation(endpoint.get("arm_results", []), "control")
            if control is not None:
                control_row, control_obs = control
                item.update(
                    {
                        "control_arm": control_row.get("arm", ""),
                        "control_count": control_obs["count"],
                        "control_denominator": control_obs["denominator"],
                        "control_rate": round(control_obs["rate"], 6),
                    }
                )
        observations[endpoint_key] = item
    return observations


def historical_endpoint_observations(
    rows: list[dict[str, Any]],
    endpoint_key: str,
) -> list[dict[str, Any]]:
    observations = []
    for item in rows:
        discount = to_float(item.get("suggested_borrowing_discount")) or 0.0
        if discount <= 0:
            continue
        for quantity in item.get("borrowable_quantities", []):
            candidate_key = canonical_bayesian_endpoint(
                quantity.get("endpoint_family", ""),
                quantity.get("endpoint", ""),
                quantity.get("time_frame", ""),
            )
            if candidate_key != endpoint_key:
                continue
            selected = select_arm_observation(quantity.get("arm_results", []), "treatment")
            if selected is None:
                continue
            row, observation = selected
            observations.append(
                {
                    "nct_id": item.get("candidate_nct_id", item.get("nct_id", "")),
                    "endpoint": quantity.get("endpoint", ""),
                    "arm": row.get("arm", ""),
                    "count": observation["count"],
                    "denominator": observation["denominator"],
                    "rate": round(observation["rate"], 6),
                    "weight": max(0.0, min(1.0, discount)),
                    "suitability": item.get("prior_borrowing_suitability", ""),
                }
            )
    return observations


def beta_grid(alpha: float, beta: float, points: int = 4001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0001, 0.9999, points)
    log_norm = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(grid) + (beta - 1.0) * np.log1p(-grid) - log_norm
    log_pdf = log_pdf - float(np.max(log_pdf))
    weights = np.exp(log_pdf)
    weights = weights / float(np.sum(weights))
    cdf = np.cumsum(weights)
    return grid, weights, cdf


def beta_probability_greater(alpha: float, beta: float, threshold: float) -> float:
    threshold = max(0.0, min(1.0, threshold))
    grid, weights, _ = beta_grid(alpha, beta)
    return float(np.sum(weights[grid >= threshold]))


def beta_quantile(alpha: float, beta: float, probability: float) -> float:
    probability = max(0.0, min(1.0, probability))
    grid, _, cdf = beta_grid(alpha, beta)
    idx = int(np.searchsorted(cdf, probability, side="left"))
    idx = min(max(idx, 0), len(grid) - 1)
    return float(grid[idx])


def beta_summary(alpha: float, beta: float) -> dict[str, Any]:
    return {
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "mean": round(alpha / (alpha + beta), 6),
        "median": round(beta_quantile(alpha, beta, 0.5), 6),
        "credible_interval_95": [
            round(beta_quantile(alpha, beta, 0.025), 6),
            round(beta_quantile(alpha, beta, 0.975), 6),
        ],
    }


def borrowing_pi_summaries(weights: list[float]) -> dict[str, float]:
    if not weights:
        return {"mean": 0.0, "top_m": 0.0, "softmax": 0.0}
    clipped = np.array([max(0.0, min(1.0, w)) for w in weights], dtype=float)
    top_m = min(5, len(clipped))
    softmax_lambda = 3.0
    scaled = np.exp(softmax_lambda * clipped)
    softmax = float(np.sum((scaled / np.sum(scaled)) * clipped))
    return {
        "mean": round(float(np.mean(clipped)), 6),
        "top_m": round(float(np.mean(np.sort(clipped)[-top_m:])), 6),
        "softmax": round(softmax, 6),
    }


def posterior_for_weight_multiplier(
    historical: list[dict[str, Any]],
    query: dict[str, Any],
    multiplier: float,
) -> dict[str, Any]:
    alpha = 1.0
    beta = 1.0
    effective_sample_size = 0.0
    weighted_events = 0.0
    for item in historical:
        weight = max(0.0, min(1.0, item["weight"] * multiplier))
        alpha += weight * item["count"]
        beta += weight * (item["denominator"] - item["count"])
        effective_sample_size += weight * item["denominator"]
        weighted_events += weight * item["count"]
    prior = beta_summary(alpha, beta)
    query_count = to_float(query.get("treatment_count"))
    query_denominator = to_float(query.get("treatment_denominator"))
    has_query_result = query_count is not None and query_denominator is not None
    if query_count is not None and query_denominator is not None:
        alpha += query_count
        beta += query_denominator - query_count
    posterior = beta_summary(alpha, beta) if has_query_result else None
    active_summary = posterior or prior
    thresholds = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
    probability_grid = [
        {
            "threshold": float(threshold),
            "posterior_probability": round(beta_probability_greater(alpha, beta, threshold), 6),
        }
        for threshold in thresholds
    ]
    tipping_points = [
        {
            "posterior_probability_level": level,
            "rate_threshold": round(beta_quantile(active_summary["alpha"], active_summary["beta"], 1.0 - level), 6),
        }
        for level in (0.5, 0.7, 0.8, 0.9)
    ]
    return {
        "effective_sample_size": round(effective_sample_size, 4),
        "weighted_events": round(weighted_events, 4),
        "weighted_rate": round(weighted_events / effective_sample_size, 6) if effective_sample_size > 0 else None,
        "prior": prior,
        "posterior": posterior,
        "active_distribution": "posterior" if has_query_result else "prior",
        "success_probability_grid": probability_grid,
        "tipping_points": tipping_points,
    }


def two_arm_orr_support(endpoint_analysis: dict[str, Any], query: dict[str, Any]) -> dict[str, Any] | None:
    control_count = to_float(query.get("control_count"))
    control_denominator = to_float(query.get("control_denominator"))
    if control_count is None or control_denominator is None or control_denominator <= 0:
        return None
    observed = next(
        (row for row in endpoint_analysis.get("weight_sensitivity", []) if row.get("scenario") == "observed_weights"),
        None,
    )
    if observed is None or observed.get("posterior") is None:
        return None
    treatment_alpha = float(observed["posterior"]["alpha"])
    treatment_beta = float(observed["posterior"]["beta"])
    control_alpha = 1.0 + control_count
    control_beta = 1.0 + control_denominator - control_count
    rng = np.random.default_rng(20260519)
    treatment_samples = rng.beta(treatment_alpha, treatment_beta, size=20000)
    control_samples = rng.beta(control_alpha, control_beta, size=20000)
    odds_ratio = (treatment_samples / (1.0 - treatment_samples)) / (
        control_samples / (1.0 - control_samples)
    )
    return {
        "model": "path_a_treatment_absolute_rate_prior_control_weak_prior",
        "control_arm": query.get("control_arm", ""),
        "control_posterior": beta_summary(control_alpha, control_beta),
        "posterior_or_mean": round(float(np.mean(odds_ratio)), 6),
        "posterior_or_median": round(float(np.quantile(odds_ratio, 0.5)), 6),
        "posterior_or_credible_interval_95": [
            round(float(np.quantile(odds_ratio, 0.025)), 6),
            round(float(np.quantile(odds_ratio, 0.975)), 6),
        ],
        "probability_grid": [
            {
                "or_threshold": threshold,
                "posterior_probability": round(float(np.mean(odds_ratio >= threshold)), 6),
            }
            for threshold in (1.0, 1.25, 1.5, 2.0, 3.0)
        ],
        "note": "OR thresholds are reference grid values only; user-defined OR_target and gamma are required for go/no-go.",
    }


def add_bayesian_analysis(result: dict[str, Any]) -> dict[str, Any]:
    query_summary = result.get("query_summary", result.get("query", {}))
    rows = result.get("reranked_top_matches") or result.get("reranked_top10") or result.get("top10") or []
    query_observations = query_endpoint_observations(query_summary)
    endpoint_analyses = []
    two_arm: dict[str, Any] = {}

    for endpoint_key, query in query_observations.items():
        historical = historical_endpoint_observations(rows, endpoint_key)
        if not historical:
            continue
        weights = [item["weight"] for item in historical]
        scenarios = [
            ("no_borrowing", 0.0),
            ("25pct_weights", 0.25),
            ("50pct_weights", 0.50),
            ("75pct_weights", 0.75),
            ("observed_weights", 1.0),
            ("125pct_capped_weights", 1.25),
        ]
        sensitivity = []
        observed_summary: dict[str, Any] | None = None
        for label, multiplier in scenarios:
            summary = posterior_for_weight_multiplier(historical, query, multiplier)
            row = {"scenario": label, "weight_multiplier": multiplier, **summary}
            sensitivity.append(row)
            if label == "observed_weights":
                observed_summary = summary
        if observed_summary is None:
            continue
        has_query_result = to_float(query.get("treatment_count")) is not None and to_float(query.get("treatment_denominator")) is not None
        analysis = {
            "endpoint_family": endpoint_key,
            "analysis_mode": "posterior" if has_query_result else "prior_only",
            "query_endpoint": query,
            "historical_observations": historical,
            "historical_trial_count": len(historical),
            "effective_sample_size": observed_summary["effective_sample_size"],
            "weighted_historical_rate": observed_summary["weighted_rate"],
            "mixture_prior": (
                mixture_prior.components_from_reranked_rows(rows, endpoint_key, lambda0=0.2)
                if mixture_prior is not None
                else {"lambda_0": 1.0, "components": []}
            ),
            "mixture_weight_pi": borrowing_pi_summaries(weights),
            "weight_sensitivity": sensitivity,
            "success_probability_grid": observed_summary["success_probability_grid"],
            "tipping_points": observed_summary["tipping_points"],
            "notes": [
                "Targets and gamma are user-defined and are not set by this analysis.",
                "This implementation uses a lightweight weighted beta-binomial power-prior approximation.",
            ],
        }
        endpoint_analyses.append(analysis)
        if endpoint_key == "ORR":
            or_support = two_arm_orr_support(analysis, query)
            if or_support is not None:
                two_arm["orr"] = or_support

    result["bayesian_analysis"] = {
        "status": "available" if endpoint_analyses else "not_available",
        "model": "weighted_beta_binomial_path_a",
        "endpoint_analyses": endpoint_analyses,
        "two_arm_decision_support": two_arm,
        "limitations": [
            "PFS is analyzed only when a 6-month count/denominator endpoint is available.",
            "Two-arm path A borrows treatment-arm absolute-rate information; OR/HR decision thresholds remain user-defined.",
            "Mixture-weight pi summaries are sensitivity descriptors; the current posterior is not a full robust MAP mixture posterior.",
        ],
    }
    return result


def render_markdown_report(result: dict[str, Any], max_rows: int = 10) -> str:
    query = result.get("query_summary", {})
    rows = result.get("reranked_top_matches") or result.get("top10", [])
    lines = [
        "# Oncology Trial Similarity Report",
        "",
        f"Query: {query.get('nct_id', 'QUERY')} - {query.get('brief_title', '')}",
        "",
        "## Query Summary",
        "",
        f"- Phase: {query.get('phase', '')}",
        f"- Cancer: {clean_text(query.get('cancer_type', {}))}",
        f"- Intervention: {clean_text(query.get('intervention', {}))}",
        f"- Design: {clean_text(query.get('design', {}))}",
        "",
        "## Reranked Top Matches",
        "",
        "| Rank | NCT ID | Score | Suitability | Discount | Key rationale | Red flags |",
        "|---:|---|---:|---|---:|---|---|",
    ]
    for idx, item in enumerate(rows[:max_rows], start=1):
        rank = item.get("rank", idx)
        score = item.get("overall_similarity_score", item.get("score_0_100", 0))
        suitability = item.get("prior_borrowing_suitability", "candidate")
        discount = item.get("suggested_borrowing_discount", "")
        rationale = clean_text(item.get("explanation", item.get("title", "")))[:180]
        flags = clean_text(item.get("red_flags", item.get("nonborrowability_risks", [])))[:180]
        lines.append(
            f"| {rank} | {item.get('candidate_nct_id', item.get('nct_id', ''))} "
            f"| {score} | {suitability} | {discount} | {rationale} | {flags} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Scores are generated by a deterministic structured reranker and should be reviewed before use in a primary Bayesian analysis.")
    lines.append("- `suggested_borrowing_discount` is a starting value for power-prior/commensurate-prior sensitivity analysis, not an automatic final borrowing weight.")
    bayesian = result.get("bayesian_analysis", {})
    endpoint_analyses = bayesian.get("endpoint_analyses", [])
    if endpoint_analyses:
        lines.append("")
        lines.append("## Bayesian Prior Borrowing Analysis")
        lines.append("")
        lines.append("- Targets and gamma are user-defined; this report does not make a go/no-go call.")
        for endpoint in endpoint_analyses:
            observed = next(
                (row for row in endpoint.get("weight_sensitivity", []) if row.get("scenario") == "observed_weights"),
                {},
            )
            posterior = observed.get("posterior") or observed.get("prior", {})
            distribution_label = "Posterior" if observed.get("posterior") is not None else "Prior-only"
            lines.append("")
            lines.append(f"### {endpoint.get('endpoint_family', '')}")
            lines.append("")
            lines.append(f"- Analysis mode: {endpoint.get('analysis_mode', '')}")
            lines.append(f"- Historical trials used: {endpoint.get('historical_trial_count', 0)}")
            lines.append(f"- Effective sample size: {endpoint.get('effective_sample_size', 0)}")
            lines.append(f"- Weighted historical rate: {endpoint.get('weighted_historical_rate')}")
            lines.append(f"- {distribution_label} mean: {posterior.get('mean')}")
            lines.append(f"- {distribution_label} 95% credible interval: {posterior.get('credible_interval_95')}")
            lines.append("")
            lines.append("| Scenario | ESS | Active distribution | Mean | 95% CI |")
            lines.append("|---|---:|---|---:|---|")
            for row in endpoint.get("weight_sensitivity", []):
                row_posterior = row.get("posterior") or row.get("prior", {})
                lines.append(
                    f"| {row.get('scenario', '')} | {row.get('effective_sample_size', 0)} "
                    f"| {row.get('active_distribution', '')} | {row_posterior.get('mean')} | {row_posterior.get('credible_interval_95')} |"
                )
    return "\n".join(lines) + "\n"


def search(
    query_json: Path,
    index_dir: Path,
    top_k: int,
    rerank_top_n: int = 0,
    embedding_backend: str | None = None,
    embedding_model: str | None = None,
    embedding_batch_size: int = 16,
    embedding_max_length: int = 256,
    retrieval_backend: str = DEFAULT_RETRIEVAL_BACKEND,
    trial2vec_index_path: Path | None = None,
    trial2vec_model_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_supported_retrieval_backend(retrieval_backend)
    if not query_json.exists():
        raise FileNotFoundError(f"Query JSON does not exist: {query_json}")
    embeddings_path = index_dir / "trial_embeddings.npz"
    summaries_path = index_dir / "trial_summaries.jsonl"
    if retrieval_backend == "clinicalbert" and (not embeddings_path.exists() or not summaries_path.exists()):
        raise FileNotFoundError(
            "Index files were not found. Run build-index first, or pass the correct --index-dir. "
            f"Expected: {embeddings_path} and {summaries_path}"
        )
    if retrieval_backend == "trial2vec":
        if trial2vec_index_path is None:
            raise ValueError("--trial2vec-index-path is required when --retrieval-backend=trial2vec.")
        if trial2vec_model_dir is None:
            raise ValueError("--trial2vec-model-dir is required when --retrieval-backend=trial2vec.")
        if not summaries_path.exists():
            raise FileNotFoundError(
                "Trial summaries were not found. Run build-index first, or pass the correct --index-dir. "
                f"Expected: {summaries_path}"
            )
        if not trial2vec_index_path.exists():
            raise FileNotFoundError(f"Trial2Vec index file does not exist: {trial2vec_index_path}")
        if not trial2vec_model_dir.exists():
            raise FileNotFoundError(f"Trial2Vec model directory does not exist: {trial2vec_model_dir}")

    tmp_folder = query_json.parent
    raw = read_json(query_json)
    folder_name = clean_text(get_nested(raw, "Study details", "1. NCT number", default="QUERY")) or tmp_folder.name
    record = TrialRecord(
        nct_id=folder_name,
        folder=tmp_folder,
        json_path=query_json,
        protocol_path=find_supporting_pdfs(tmp_folder)["protocol"],
        sap_path=find_supporting_pdfs(tmp_folder)["sap"],
        raw_json=raw,
        extracted=extract_trial_record_like(raw, folder_name, query_json),
    )
    query_summary = make_rule_based_summary(record.extracted)

    summaries = load_summaries(summaries_path)
    scored = []
    result_embedding_backend = ""
    result_embedding_model = ""

    if retrieval_backend == "clinicalbert":
        embeddings_file = np.load(embeddings_path, allow_pickle=False)
        stored_backend = str(embeddings_file["embedding_backend"][0]) if "embedding_backend" in embeddings_file else "hashing"
        stored_model = str(embeddings_file["embedding_model"][0]) if "embedding_model" in embeddings_file else "signed-token-hashing-2048"
        active_backend = embedding_backend or stored_backend
        active_model = embedding_model or (stored_model if active_backend != "hashing" else DEFAULT_CLINICALBERT_MODEL)
        if active_backend != stored_backend:
            raise ValueError(
                f"Query embedding backend ({active_backend}) does not match index backend ({stored_backend}). "
                "Rebuild the index with the desired backend or pass the matching --embedding-backend."
            )
        query_embedder = make_embedder(
            active_backend,
            model_name=active_model,
            batch_size=embedding_batch_size,
            max_length=embedding_max_length,
        )
        query_emb = summary_embedding(query_summary, query_embedder)
        embeddings = {
            "nct_ids": embeddings_file["nct_ids"],
            **{aspect: embeddings_file[aspect] for aspect in ASPECT_WEIGHTS},
        }
        nct_ids = embeddings["nct_ids"]

        for idx, nct_id_raw in enumerate(nct_ids):
            nct_id = str(nct_id_raw)
            if nct_id == query_summary.get("nct_id"):
                continue
            by_aspect = {}
            score = 0.0
            for aspect, weight in ASPECT_WEIGHTS.items():
                sim = cosine(query_emb[aspect], embeddings[aspect][idx])
                by_aspect[aspect] = sim
                score += weight * sim
            scored.append(
                enrich_candidate_row(
                    {
                        "nct_id": nct_id,
                        "score": score,
                        "score_0_100": round(100 * max(0.0, score), 2),
                        "aspect_scores": {k: round(v, 4) for k, v in by_aspect.items()},
                        "retrieval_backend": "clinicalbert",
                    },
                    summaries.get(nct_id, {}),
                )
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        result_embedding_backend = stored_backend
        result_embedding_model = stored_model
    elif retrieval_backend == "trial2vec":
        assert trial2vec_index_path is not None
        assert trial2vec_model_dir is not None
        query_row = summary_to_trial2vec_row(query_summary)
        query_vector = encode_trial2vec_query(query_row, trial2vec_model_dir)
        scored = score_trial2vec_index(
            query_vector,
            trial2vec_index_path,
            excluded_nct_id=query_summary.get("nct_id", ""),
            top_k=max(rerank_top_n, top_k),
        )
        scored = [
            enrich_candidate_row(row, summaries.get(row["nct_id"], {}))
            for row in scored
        ]
        result_embedding_backend = "trial2vec"
        result_embedding_model = str(trial2vec_model_dir)

    top_matches = scored[:top_k]
    result = {
        "query_summary": query_summary,
        "retrieval_backend": retrieval_backend,
        "embedding_backend": result_embedding_backend,
        "embedding_model": result_embedding_model,
        "top_matches": top_matches,
        "top10": top_matches[: min(top_k, 10)],
    }
    if rerank_top_n > 0:
        rerank_input = scored[: max(rerank_top_n, top_k)]
        reranked = rerank_candidates(query_summary, rerank_input, rerank_top_n)
        result["reranked_top_matches"] = reranked
        result["reranked_top10"] = reranked[:10]
    return add_bayesian_analysis(result)


def extract_trial_record_like(raw: dict[str, Any], fallback_nct_id: str, json_path: Path) -> dict[str, Any]:
    details = raw.get("Study details", {})
    results = raw.get("Results Posted", {})
    overview = get_nested(details, "5. Study Overview", default={})
    design = get_nested(results, "2. Study Design", default={})
    dates = get_nested(results, "4. Study Record Dates", default={})
    outcomes = extract_outcomes(results if isinstance(results, dict) else {})
    nct_id = clean_text(details.get("1. NCT number")) or fallback_nct_id
    pdfs = find_supporting_pdfs(json_path.parent)
    return {
        "nct_id": nct_id,
        "json_path": str(json_path),
        "brief_title": clean_text(overview.get("Brief Title") if isinstance(overview, dict) else ""),
        "official_title": clean_text(overview.get("Official Title") if isinstance(overview, dict) else ""),
        "brief_summary": clean_text(overview.get("Brief Summary") if isinstance(overview, dict) else ""),
        "detailed_description": clean_text(overview.get("Detailed Description") if isinstance(overview, dict) else ""),
        "status": clean_text(details.get("2. Study status") if isinstance(details, dict) else ""),
        "phase": clean_text(details.get("7. Phase") if isinstance(details, dict) else ""),
        "interventions": results.get("1. Intervention/Treatment", []) if isinstance(results, dict) else [],
        "design": design if isinstance(design, dict) else {},
        "enrollment": clean_text(results.get("3. Enrollment (Actual)") if isinstance(results, dict) else ""),
        "dates": dates if isinstance(dates, dict) else {},
        "supporting_documents": {
            "protocol_pdf": str(pdfs["protocol"]) if pdfs["protocol"] else "",
            "sap_pdf": str(pdfs["sap"]) if pdfs["sap"] else "",
            "protocol_excerpt": read_pdf_excerpt(pdfs["protocol"], max_chars=8000),
            "sap_excerpt": read_pdf_excerpt(pdfs["sap"], max_chars=5000),
        },
        "outcomes": [{**o, "endpoint_family": infer_endpoint_family(o["title"])} for o in outcomes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and search an oncology clinical trial similarity index."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index")
    build.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    build.add_argument("--output-dir", type=Path, default=Path("artifacts/oncology_trial_similarity"))
    build.add_argument(
        "--embedding-backend",
        choices=["hashing", "clinicalbert"],
        default="hashing",
        help="Embedding backend for index construction. clinicalbert uses cached Bio_ClinicalBERT via transformers.",
    )
    build.add_argument("--embedding-model", default=DEFAULT_CLINICALBERT_MODEL)
    build.add_argument("--embedding-batch-size", type=int, default=16)
    build.add_argument("--embedding-max-length", type=int, default=256)

    query = sub.add_parser("search")
    query.add_argument("--query-json", type=Path, required=True)
    query.add_argument("--index-dir", type=Path, default=Path("artifacts/oncology_trial_similarity"))
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument(
        "--embedding-backend",
        choices=["hashing", "clinicalbert"],
        default=None,
        help="Optional query embedding backend. Defaults to the backend stored in the index.",
    )
    query.add_argument("--embedding-model", default=None)
    query.add_argument("--embedding-batch-size", type=int, default=16)
    query.add_argument("--embedding-max-length", type=int, default=256)
    query.add_argument(
        "--retrieval-backend",
        choices=list(RETRIEVAL_BACKENDS),
        default=DEFAULT_RETRIEVAL_BACKEND,
    )
    query.add_argument("--trial2vec-index-path", type=Path, default=None)
    query.add_argument("--trial2vec-model-dir", type=Path, default=None)
    query.add_argument(
        "--rerank",
        action="store_true",
        help="Run deterministic prior-borrowing rerank over the first-stage candidates.",
    )
    query.add_argument(
        "--rerank-top-n",
        type=int,
        default=100,
        help="Number of first-stage candidates to rerank when --rerank is used.",
    )
    query.add_argument("--output", type=Path, default=None)
    query.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional Markdown report path for reranked or first-stage results.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        print(
            "\nExamples:\n"
            "  Build the local trial index:\n"
            "    python3 oncology_trial_similarity_pipeline.py build-index\n\n"
            "  Search similar trials for one query JSON:\n"
            "    python3 oncology_trial_similarity_pipeline.py search "
            "--query-json /path/to/new_trial.json "
            "--output artifacts/oncology_trial_similarity/new_trial_top10.json\n"
        )
        return

    args = parser.parse_args()
    if args.command == "build-index":
        build_index(
            args.db_root,
            args.output_dir,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_length=args.embedding_max_length,
        )
    elif args.command == "search":
        rerank_top_n = args.rerank_top_n if args.rerank else 0
        result = search(
            args.query_json,
            args.index_dir,
            args.top_k,
            rerank_top_n,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_length=args.embedding_max_length,
            retrieval_backend=args.retrieval_backend,
            trial2vec_index_path=args.trial2vec_index_path,
            trial2vec_model_dir=args.trial2vec_model_dir,
        )
        text = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(text)
        if args.report_output:
            args.report_output.parent.mkdir(parents=True, exist_ok=True)
            args.report_output.write_text(render_markdown_report(result), encoding="utf-8")
            print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
