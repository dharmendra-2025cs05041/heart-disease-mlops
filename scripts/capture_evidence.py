"""
Render verification command outputs into PNG evidence screenshots.

The assignment rubric (Task 7 + Task 8) explicitly asks for deployment
and monitoring screenshots. This script captures the live state of the
running stack and writes one PNG per artefact into screenshots/.

Run this AFTER `bash scripts/deploy.sh kubernetes` and the docker
compose monitoring stack are both up.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parent.parent
SHOTS = REPO / "screenshots"
SHOTS.mkdir(exist_ok=True)


def run(cmd: str, env_path_extra: str = "") -> str:
    """Run a shell command and return its combined stdout/stderr."""
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception as exc:
        return f"<error: {exc}>"


def render_terminal(title: str, body: str, out_name: str) -> Path:
    """Render text as a dark-themed terminal-style PNG."""
    lines = body.rstrip().splitlines() or ["<no output>"]
    width = max(min(max(len(line) for line in lines), 130), 60)
    height = max(len(lines) + 4, 8)

    fig_w = max(width * 0.085, 8)
    fig_h = max(height * 0.22, 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")

    mono = next(
        (f for f in ["Menlo", "Consolas", "DejaVu Sans Mono", "Courier New"]
         if any(f.lower() in fn.name.lower() for fn in font_manager.fontManager.ttflist)),
        "monospace",
    )

    ax.text(
        0.01, 0.97,
        f"$ {title}",
        family=mono, fontsize=10, color="#7ec699",
        ha="left", va="top", transform=ax.transAxes,
    )
    ax.text(
        0.01, 0.92,
        "\n".join(lines),
        family=mono, fontsize=9, color="#d4d4d4",
        ha="left", va="top", transform=ax.transAxes,
    )

    out = SHOTS / out_name
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")
    return out


def main() -> None:
    captures = [
        ("kubectl get pods,svc,hpa -l app=heart-disease-api",
         "k8s_deployment_state.png"),
        ("kubectl describe hpa heart-disease-api-hpa",
         "k8s_hpa_metrics.png"),
        ("kubectl top pods -l app=heart-disease-api",
         "k8s_pods_resource_usage.png"),
        ("/Users/dparsail/.rd/bin/docker compose -f deployment/docker/docker-compose.yml ps",
         "docker_compose_stack.png"),
    ]

    print("Capturing kubectl + docker outputs...")
    for cmd, out_name in captures:
        text = run(cmd)
        render_terminal(cmd, text, out_name)

    print("Capturing Prometheus targets (JSON -> table)...")
    raw = run("curl -fsS http://localhost:9090/api/v1/targets?state=active")
    try:
        data = json.loads(raw)
        rows = ["JOB                    HEALTH   SCRAPE URL"]
        rows.append("-" * 70)
        for t in data["data"]["activeTargets"]:
            rows.append(
                f"{t['labels']['job']:<22} {t['health']:<8} {t['scrapeUrl']}"
            )
        body = "\n".join(rows)
    except Exception:
        body = raw[:2000]
    render_terminal(
        "curl http://localhost:9090/api/v1/targets",
        body,
        "prometheus_targets.png",
    )

    print("Capturing /predict response (proof the API serves predictions)...")
    sample = (REPO / "scripts" / "sample_payload.json").read_text()
    cmd = (
        "curl -fsS -X POST http://localhost:8000/predict "
        "-H 'Content-Type: application/json' "
        f"-d {shlex.quote(sample)}"
    )
    raw = run(cmd)
    try:
        body = json.dumps(json.loads(raw), indent=2)
    except Exception:
        body = raw
    render_terminal(
        "curl -X POST http://localhost:8000/predict",
        body,
        "api_predict_response.png",
    )

    print("\nDone. Files written to:", SHOTS.relative_to(REPO))


if __name__ == "__main__":
    main()
