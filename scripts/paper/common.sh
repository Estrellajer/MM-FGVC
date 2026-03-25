#!/usr/bin/env bash

paper_notify() {
  local status="$1"
  local suite_name="$2"
  local timestamp="$3"
  local models="$4"
  local seeds="$5"
  local task_groups="$6"
  local method_groups="$7"
  local summary_path="$8"
  local extra_paths="${9:-}"

  if [[ -z "${FEISHU_WEBHOOK_URL:-}" ]]; then
    return 0
  fi

  local host
  host="$(hostname)"

  local body
  body=$(
    cat <<EOF
status=${status}
suite=${suite_name}
timestamp=${timestamp}
host=${host}
models=${models}
seeds=${seeds}
task_groups=${task_groups}
method_groups=${method_groups}
summary=${summary_path}
EOF
  )

  if [[ -n "${extra_paths}" ]]; then
    body="${body}"$'\n'"extra=${extra_paths}"
  fi

  uv run python scripts/paper/notify_feishu.py \
    --webhook "${FEISHU_WEBHOOK_URL}" \
    --title "[FGVC] ${suite_name} ${status}" \
    --body "${body}" || true
}


paper_notify_on_exit() {
  local exit_code="$1"
  local suite_name="$2"
  local timestamp="$3"
  local models="$4"
  local seeds="$5"
  local task_groups="$6"
  local method_groups="$7"
  local summary_path="$8"
  local extra_paths="${9:-}"

  local status="SUCCESS"
  if [[ "${exit_code}" != "0" ]]; then
    status="FAILED(${exit_code})"
  fi

  paper_notify \
    "${status}" \
    "${suite_name}" \
    "${timestamp}" \
    "${models}" \
    "${seeds}" \
    "${task_groups}" \
    "${method_groups}" \
    "${summary_path}" \
    "${extra_paths}"
}
