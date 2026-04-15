import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path:str) -> Optional[Dict[str, Any]]:
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def analyze_pytest(report: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "failed_tests": [],
        "slowest_tests": []
    }
    tests = report.get("tests", [])
    result["total"] = len(tests)
    durations = []
    for test in tests:
        name = test.get("nodeid")
        outcome = test.get("outcome")
        duration = test.get("call", {}).get("duration", 0)
        durations.append((name, duration))
        if outcome == "passed":
            result["passed"] += 1
        elif outcome == "failed":
            result["failed"] += 1
            result["failed_tests"].append(name)
    durations.sort(key=lambda x: x[1], reverse=True)
    result["slowest_tests"] = durations[:10]
    return result


def analyze_coverage(cov: Dict[str, Any]) -> Dict[str, Any]:
    if not cov:
        return {"coverage_total": None}
    return {
        "coverage_total": cov.get("totals", {}).get("percent_covered", None)
    }


def main():
    pytest_report = load_json("report.json")
    coverage_report = load_json("coverage.json")
    analysis = {
        "pytest": analyze_pytest(pytest_report) if pytest_report else {},
        "coverage": analyze_coverage(coverage_report)
    }
    with open("test_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)


if __name__ == "__main__":
    main()
