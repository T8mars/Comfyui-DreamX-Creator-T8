import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SOURCE_ROOTS = (
    ROOT / "comfy_nodes",
    ROOT / "audio_video_generation",
    ROOT / "video_refiner",
    ROOT / "scripts",
)
IGNORED_FILES = {
    "scripts/render_demo.py",
    "audio_video_generation/inference.py",
    "audio_video_generation/inference_sp.py",
    "video_refiner/inference_sr.py",
}
KNOWN_SCANNER_TRIGGERS = {
    "environment access": re.compile(r"os\.(?:environ\b|getenv\s*\()"),
    "dynamic import": re.compile(r"importlib\.import_module\s*\("),
    "command execution": re.compile(
        r"subprocess\.(?:Popen|run|call|check_output|check_call)\s*\(|os\.system\s*\("
    ),
    "raw network client": re.compile(
        r"requests\.(?:get|post|put|patch|delete|head)\s*\(|"
        r"urllib\.request\.urlopen\s*\(|aiohttp\s*\.\s*ClientSession|"
        r"socket\.socket\s*\("
    ),
}


def test_registry_shipped_python_avoids_known_false_positive_patterns():
    hits = []
    paths = [ROOT / "__init__.py"]
    for source_root in SHIPPED_SOURCE_ROOTS:
        paths.extend(source_root.rglob("*.py"))
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative in IGNORED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for label, trigger in KNOWN_SCANNER_TRIGGERS.items():
            if trigger.search(text):
                hits.append(f"{relative}: {label}")
    assert not hits, "Registry scanner triggers found:\n" + "\n".join(hits)
