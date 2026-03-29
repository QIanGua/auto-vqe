#!/bin/bash
# LiH Autoresearch Benchmark Entrypoint
# This script runs exactly one strategy per iteration, verifies its candidate,
# then emits machine-readable METRIC lines for the outer research driver.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

REQUESTED_STRATEGY="${1:-ga}"
if [ "$REQUESTED_STRATEGY" != "ga" ] && [ "$REQUESTED_STRATEGY" != "multidim" ]; then
    echo "Unknown strategy: $REQUESTED_STRATEGY"
    echo "Expected one of: ga, multidim"
    exit 2
fi

SESSION_DIR="${AGENT_VQE_SESSION_DIR:-}"
ITERATION_ID="${AGENT_VQE_ITERATION:-manual}"
if [ -n "$SESSION_DIR" ]; then
    ITER_DIR="$SESSION_DIR/iterations/${ITERATION_ID}_${REQUESTED_STRATEGY}"
    mkdir -p "$ITER_DIR"
else
    TMP_DIR="$(mktemp -d)"
    ITER_DIR="$TMP_DIR"
    trap 'rm -rf "$TMP_DIR"' EXIT
fi

run_python() {
    uv run python "$@"
}

extract_metric() {
    local file="$1"
    local key="$2"
    grep -a "^${key}[[:space:]]*:" "$file" | tail -n 1 | awk -F': ' '{print $2}'
}

verify_config() {
    local strategy="$1"
    local config_path="$2"
    local out_file="$3"

    echo "Verifying ${strategy} candidate from ${config_path} ..."
    if ! run_python experiments/lih/run.py --config "$config_path" --trials 2 >"$out_file" 2>&1; then
        echo "Verification failed for ${strategy}."
        cat "$out_file"
        return 1
    fi

    local val_energy
    local energy_error
    local num_params
    val_energy="$(extract_metric "$out_file" "val_energy")"
    energy_error="$(extract_metric "$out_file" "energy_error")"
    num_params="$(extract_metric "$out_file" "num_params")"

    if [ -z "$val_energy" ] || [ -z "$energy_error" ] || [ -z "$num_params" ]; then
        echo "Failed to parse verification metrics for ${strategy}."
        cat "$out_file"
        return 1
    fi

    printf '%s\n' "$val_energy" "$energy_error" "$num_params"
    return 0
}

SEARCH_LOG="$ITER_DIR/${REQUESTED_STRATEGY}_search.log"
VERIFY_LOG="$ITER_DIR/${REQUESTED_STRATEGY}_verify.log"

if [ "$REQUESTED_STRATEGY" = "ga" ]; then
    SEARCH_ENTRY="experiments/lih/ga/search.py"
    if [ -n "$SESSION_DIR" ]; then
        CONFIG_PATH="$SESSION_DIR/ga/best_config_ga.json"
    else
        CONFIG_PATH="$REPO_ROOT/experiments/lih/ga/best_config_ga.json"
    fi
else
    SEARCH_ENTRY="experiments/lih/multidim/search.py"
    if [ -n "$SESSION_DIR" ]; then
        CONFIG_PATH="$SESSION_DIR/multidim/best_config_multidim.json"
    else
        CONFIG_PATH="$REPO_ROOT/experiments/lih/multidim/best_config_multidim.json"
    fi
fi

echo "Running LiH ${REQUESTED_STRATEGY} Search Phase..."
if run_python "$SEARCH_ENTRY" >"$SEARCH_LOG" 2>&1; then
    SEARCH_OK=1
else
    SEARCH_OK=0
    echo "${REQUESTED_STRATEGY} search failed."
    cat "$SEARCH_LOG"
fi

if [ "${SEARCH_OK:-0}" -ne 1 ] || [ ! -f "$CONFIG_PATH" ]; then
    echo "${REQUESTED_STRATEGY} did not produce a verifiable config."
    exit 1
fi

mapfile -t METRICS < <(verify_config "$REQUESTED_STRATEGY" "$CONFIG_PATH" "$VERIFY_LOG")
if [ "${#METRICS[@]}" -ne 3 ]; then
    echo "Verification metrics missing for ${REQUESTED_STRATEGY}."
    exit 1
fi

echo "--- METRICS ---"
echo "METRIC val_energy=${METRICS[0]}"
echo "METRIC energy_error=${METRICS[1]}"
echo "METRIC num_params=${METRICS[2]}"
echo "METRIC selected_strategy=$REQUESTED_STRATEGY"
echo "METRIC selected_config_path=$CONFIG_PATH"
echo "---------------"
