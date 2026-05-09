#!/usr/bin/env bash
# One-shot demo stack runner: api + prometheus + grafana via docker compose.
# Idempotent: safe to re-run.
# Usage:
#   bash scripts/run_demo_stack.sh                 # rebuild api, up stack, smoke + warm-up burst, print URLs
#   bash scripts/run_demo_stack.sh --no-rebuild    # skip --no-cache rebuild (faster if src/ unchanged)
#   bash scripts/run_demo_stack.sh --burst-only    # assume stack is up; just fire 30+30 traffic burst
#   bash scripts/run_demo_stack.sh --status        # show stack status + URLs, no traffic
#   bash scripts/run_demo_stack.sh --down          # tear stack down (keep volumes)
#   bash scripts/run_demo_stack.sh --down --purge  # tear stack down AND wipe prom/grafana volumes

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="deployment/docker/docker-compose.yml"
API_URL="http://localhost:8000"
PROM_URL="http://localhost:9090"
GRAF_URL="http://localhost:3000"
HIGH_PAYLOAD="scripts/sample_payload_high_risk.json"
LOW_PAYLOAD="scripts/sample_payload_low_risk.json"
BURST_N=30

REBUILD=1
MODE="full"
PURGE=0
for a in "$@"; do
  case "$a" in
    --no-rebuild) REBUILD=0 ;;
    --burst-only) MODE="burst" ;;
    --status)     MODE="status" ;;
    --down)       MODE="down" ;;
    --purge)      PURGE=1 ;;
    -h|--help)    sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag: $a (try --help)"; exit 1 ;;
  esac
done

say()  { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
ok()   { printf "    \033[1;32m\xe2\x9c\x93\033[0m %s\n" "$*"; }
warn() { printf "    \033[1;33m!\033[0m %s\n" "$*"; }
die()  { printf "    \033[1;31m\xe2\x9c\x97 %s\033[0m\n" "$*" >&2; exit 1; }

dc() { docker compose -f "$COMPOSE_FILE" "$@"; }

preflight() {
  command -v docker >/dev/null 2>&1 || die "docker CLI not found"
  if ! docker info >/dev/null 2>&1; then
    local ctx; ctx="$(docker context show 2>/dev/null || echo unknown)"
    printf "    \033[1;31m\xe2\x9c\x97 docker engine not reachable (active context: %s)\033[0m\n" "$ctx" >&2
    printf "      \xe2\x86\xb3 If Rancher Desktop:  open -a 'Rancher Desktop'  (then wait for the whale icon to settle; ensure Container Engine = dockerd (moby), not containerd)\n" >&2
    printf "      \xe2\x86\xb3 If Docker Desktop :  open -a Docker\n" >&2
    printf "      \xe2\x86\xb3 If Colima         :  colima start\n" >&2
    printf "      \xe2\x86\xb3 Verify with        :  docker info | head -5\n" >&2
    exit 1
  fi
  [ -f "$COMPOSE_FILE" ]             || die "compose file missing: $COMPOSE_FILE"
  [ -f "$HIGH_PAYLOAD" ]             || die "missing $HIGH_PAYLOAD"
  [ -f "$LOW_PAYLOAD" ]              || die "missing $LOW_PAYLOAD"
}

free_ports() {
  local pids
  pids="$(lsof -ti tcp:8000 tcp:9090 tcp:3000 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    warn "killing host processes holding 8000/9090/3000: $pids"
    kill -9 $pids 2>/dev/null || true
  fi
}

wait_healthy() {
  say "Waiting for API to report /health (max 60s)"
  for i in $(seq 1 30); do
    if curl -fsS --max-time 2 "$API_URL/health" >/dev/null 2>&1; then
      ok "API is healthy at $API_URL"; return 0
    fi
    sleep 2
  done
  dc logs --no-color api --tail 50 || true
  die "API did not become healthy in 60s"
}

smoke_test() {
  say "Smoke-testing API"
  curl -fsS "$API_URL/health" | python3 -m json.tool
  echo "--- HIGH-risk patient (expect prediction=1) ---"
  curl -fsS -X POST "$API_URL/predict" -H 'Content-Type: application/json' -d @"$HIGH_PAYLOAD" | python3 -m json.tool
  echo "--- LOW-risk patient  (expect prediction=0) ---"
  curl -fsS -X POST "$API_URL/predict" -H 'Content-Type: application/json' -d @"$LOW_PAYLOAD"  | python3 -m json.tool
  ok "smoke test passed"
}

warm_up_burst() {
  say "Firing warm-up burst (${BURST_N} HIGH + ${BURST_N} LOW = $((BURST_N*2)) requests)"
  for _ in $(seq 1 "$BURST_N"); do
    curl -fsS -o /dev/null -X POST "$API_URL/predict" -H 'Content-Type: application/json' -d @"$HIGH_PAYLOAD"
    curl -fsS -o /dev/null -X POST "$API_URL/predict" -H 'Content-Type: application/json' -d @"$LOW_PAYLOAD"
  done
  ok "burst sent — sleeping 10s for next Prometheus scrape"
  sleep 10
}

verify_prom() {
  say "Verifying Prometheus has both class series"
  local q='predictions_by_class_total'
  local out; out="$(curl -fsS --get "$PROM_URL/api/v1/query" --data-urlencode "query=$q" || true)"
  local n; n="$(printf '%s' "$out" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']['result']))" 2>/dev/null || echo 0)"
  if [ "${n:-0}" -ge 2 ]; then ok "Prometheus has $n series for $q"
  else warn "expected >=2 series for $q, got ${n:-0} — re-run with --burst-only if Grafana shows 'No data'"
  fi
  ok "scrape targets: $PROM_URL/targets"
}

print_urls() {
  cat <<EOF

==> Endpoints
    Swagger UI (labelled examples) : $API_URL/docs
    Raw metrics                     : $API_URL/metrics
    Prometheus targets              : $PROM_URL/targets
    Prometheus graph (counter)      : $PROM_URL/graph?g0.expr=prediction_requests_total&g0.tab=1
    Prometheus graph (by class)     : $PROM_URL/graph?g0.expr=predictions_by_class_total&g0.tab=1
    Grafana (admin / admin)         : $GRAF_URL/dashboards
    Grafana dashboard direct link   : $GRAF_URL/d/heart-disease-api

    Tear down when done:
      bash scripts/run_demo_stack.sh --down
EOF
}

# --------------------------------------------------------------- modes
case "$MODE" in
  down)
    say "Bringing stack down"
    if [ "$PURGE" = "1" ]; then dc down -v --remove-orphans; ok "stack + volumes wiped"
    else dc down --remove-orphans; ok "stack stopped (volumes kept)"; fi
    exit 0 ;;
  status)
    preflight; dc ps; print_urls; exit 0 ;;
  burst)
    preflight; curl -fsS --max-time 3 "$API_URL/health" >/dev/null || die "API not reachable on $API_URL"
    warm_up_burst; verify_prom; print_urls; exit 0 ;;
esac

# --------------------------------------------------------------- full bring-up
preflight
say "Tearing down any previous stack"; dc down --remove-orphans 2>/dev/null || true
free_ports
if [ "$REBUILD" = "1" ]; then say "Rebuilding api image (--no-cache)"; dc build --no-cache api
else say "Reusing existing api image (use without --no-rebuild after editing src/)"; dc build api; fi
say "Starting api + prometheus + grafana"; dc up -d
dc ps
wait_healthy
smoke_test
warm_up_burst
verify_prom
print_urls
ok "demo stack ready"
