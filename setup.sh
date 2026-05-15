#!/usr/bin/env bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script directory (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { printf "${CYAN}  -> ${NC}%s\n" "$*"; }
ok()    { printf "${GREEN}  ✓ ${NC}%s\n" "$*"; }
warn()  { printf "${YELLOW}  ! ${NC}%s\n" "$*"; }
err()   { printf "${RED}  ✗ ${NC}%s\n" "$*" >&2; }

# Prompt with default value. Usage: prompt "Question" DEFAULT
# Sets REPLY to the user's answer (or default on empty input).
prompt() {
  local question="$1"
  local default="${2:-}"
  if [[ -n "$default" ]]; then
    printf "  ${BOLD}%s${NC} [%s]: " "$question" "$default"
  else
    printf "  ${BOLD}%s${NC}: " "$question"
  fi
  read -r REPLY
  REPLY="${REPLY:-$default}"
}

# Prompt for a choice from numbered options. Usage: choose "Question" option1 option2 ...
# Sets REPLY to the chosen option string.
choose() {
  local question="$1"
  shift
  local options=("$@")
  printf "  ${BOLD}%s${NC}\n" "$question"
  for i in "${!options[@]}"; do
    printf "    %d) %s\n" "$((i + 1))" "${options[$i]}"
  done
  while true; do
    printf "  > "
    read -r REPLY
    if [[ "$REPLY" =~ ^[0-9]+$ ]] && [[ "$REPLY" -ge 1 ]] && [[ "$REPLY" -le "${#options[@]}" ]]; then
      REPLY="${options[$((REPLY - 1))]}"
      return
    fi
    err "Please enter a number between 1 and ${#options[@]}"
  done
}

# Prompt for hidden input (passwords/keys). Usage: prompt_secret "Question"
# Sets REPLY to the user's answer.
prompt_secret() {
  local question="$1"
  printf "  ${BOLD}%s${NC}: " "$question"
  read -rs REPLY
  printf "\n"
}

# Mask a key for display: first 4 chars + "..." + last 3 chars.
# Short keys (<8 chars) show "***...***".
mask_key() {
  local key="$1"
  if [[ ${#key} -lt 8 ]]; then
    echo "***...***"
  else
    echo "${key:0:4}...${key: -3}"
  fi
}

# Ask yes/no. Returns 0 for yes, 1 for no. Usage: confirm "Question" [Y|N]
confirm() {
  local question="$1"
  local default="${2:-Y}"
  local prompt_str
  if [[ "$default" == "Y" ]]; then
    prompt_str="(Y/n)"
  else
    prompt_str="(y/N)"
  fi
  printf "  ${BOLD}%s${NC} %s " "$question" "$prompt_str"
  read -r REPLY
  REPLY="${REPLY:-$default}"
  [[ "$REPLY" =~ ^[Yy] ]]
}

# Check if a command exists.
has_cmd() {
  command -v "$1" &>/dev/null
}

# ---------------------------------------------------------------------------
# Step 1: Dependency check
# ---------------------------------------------------------------------------

check_deps() {
  info "Checking dependencies..."

  local has_go=false
  local has_docker=false

  if has_cmd go; then
    ok "Go $(go version | sed -n 's/.*go\([0-9][0-9.]*\).*/\1/p')"
    has_go=true
  else
    warn "Go not found (needed for local build)"
  fi

  if has_cmd docker && docker compose version &>/dev/null; then
    ok "Docker + Docker Compose"
    has_docker=true
  else
    warn "Docker or Docker Compose not found"
  fi

  if ! $has_go && ! $has_docker; then
    err "Neither Go nor Docker found. Install one of:"
    err "  Go:     https://go.dev/dl/"
    err "  Docker: https://docs.docker.com/get-docker/"
    exit 1
  fi

  # Store for later use by ask_install_method
  HAS_GO=$has_go
  HAS_DOCKER=$has_docker
}

# ---------------------------------------------------------------------------
# Step 2: Install method
# ---------------------------------------------------------------------------

ask_install_mode() {
  if $HAS_GO && $HAS_DOCKER; then
    choose "How do you want to run the proxy?" \
      "Build locally (go build)" \
      "Docker Compose"
    if [[ "$REPLY" == "Build locally (go build)" ]]; then
      INSTALL_MODE="build"
    else
      INSTALL_MODE="docker"
    fi
  elif $HAS_GO; then
    INSTALL_MODE="build"
    ok "Using local build (Go available, Docker not found)"
  else
    INSTALL_MODE="docker"
    ok "Using Docker Compose (Go not found)"
  fi
}

# ---------------------------------------------------------------------------
# Step 3: Backend selection
# ---------------------------------------------------------------------------

ask_backend_type() {
  choose "Which backend will the proxy connect to?" \
    "DeepSeek (native Anthropic-compatible API)" \
    "LiteLLM (OpenAI-compatible proxy — any model via LiteLLM)"
  if [[ "$REPLY" == "DeepSeek (native Anthropic-compatible API)" ]]; then
    BACKEND_TYPE="deepseek"
  else
    BACKEND_TYPE="litellm"
  fi
}

# ---------------------------------------------------------------------------
# Step 4: Backend configuration
# ---------------------------------------------------------------------------

ask_backend_config() {
  # Common: listen address
  prompt "Proxy listen address" "127.0.0.1:8082"
  LISTEN_ADDR="$REPLY"

  if [[ "$BACKEND_TYPE" == "deepseek" ]]; then
    ask_deepseek_config
  else
    ask_litellm_config
  fi
}

ask_deepseek_config() {
  prompt_secret "DeepSeek API Key"
  DEEPSEEK_API_KEY="$REPLY"
  if [[ -z "$DEEPSEEK_API_KEY" ]]; then
    err "DeepSeek API key is required"
    exit 1
  fi

  prompt "DeepSeek Base URL" "https://api.deepseek.com/anthropic"
  DEEPSEEK_BASE_URL="$REPLY"
}

ask_litellm_config() {
  prompt "LiteLLM Base URL" "http://localhost:4000/v1"
  LITELLM_BASE_URL="$REPLY"

  prompt_secret "LiteLLM API Key (leave empty for no-auth mode)"
  LITELLM_API_KEY="$REPLY"

  ask_model_map
}

ask_model_map() {
  # Model map stored as parallel plain arrays for broad bash compatibility.
  MODEL_MAP_KEYS=()
  MODEL_MAP_VALS=()

  local aliases=("sonnet" "opus" "haiku")
  local examples=("anthropic/claude-sonnet-4-20250514" "anthropic/claude-opus-4-20250514" "openai/Qwen3_Coder_Next")

  info "Model mapping: Claude model names → LiteLLM model identifiers"
  for i in "${!aliases[@]}"; do
    prompt "Map Claude \"${aliases[$i]}\" to which LiteLLM model? (e.g. ${examples[$i]})" ""
    if [[ -n "$REPLY" ]]; then
      MODEL_MAP_KEYS+=("${aliases[$i]}")
      MODEL_MAP_VALS+=("$REPLY")
    fi
  done

  # Auto-add full model IDs based on short aliases
  local full_ids=("claude-sonnet-4-20250514" "claude-opus-4-20250514" "claude-3-5-haiku-20241022")
  for i in "${!aliases[@]}"; do
    local short_val=""
    for j in "${!MODEL_MAP_KEYS[@]}"; do
      if [[ "${MODEL_MAP_KEYS[$j]}" == "${aliases[$i]}" ]]; then
        short_val="${MODEL_MAP_VALS[$j]}"
        break
      fi
    done
    if [[ -n "$short_val" ]]; then
      local already=false
      for j in "${!MODEL_MAP_KEYS[@]}"; do
        if [[ "${MODEL_MAP_KEYS[$j]}" == "${full_ids[$i]}" ]]; then
          already=true
          break
        fi
      done
      if ! $already; then
        MODEL_MAP_KEYS+=("${full_ids[$i]}")
        MODEL_MAP_VALS+=("$short_val")
      fi
    fi
  done

  # Custom mappings
  if confirm "Add more model mappings?" "N"; then
    while true; do
      prompt "  Claude model name (empty to stop)" ""
      local custom_key="$REPLY"
      [[ -z "$custom_key" ]] && break
      prompt "  LiteLLM model for \"$custom_key\"" ""
      local custom_val="$REPLY"
      if [[ -n "$custom_val" ]]; then
        MODEL_MAP_KEYS+=("$custom_key")
        MODEL_MAP_VALS+=("$custom_val")
      fi
    done
  fi
}

# ---------------------------------------------------------------------------
# Step 5: Claude wrapper configuration
# ---------------------------------------------------------------------------

ask_wrapper_config() {
  info "Claude Code wrapper settings"

  prompt "Default Claude model (sonnet/opus/haiku)" "sonnet"
  ANTHROPIC_MODEL="$REPLY"

  prompt "Subagent model (sonnet/opus/haiku)" "sonnet"
  CLAUDE_CODE_SUBAGENT_MODEL="$REPLY"
}

# ---------------------------------------------------------------------------
# Step 6: Summary and confirmation
# ---------------------------------------------------------------------------

show_summary() {
  printf "\n${BOLD}=== Configuration Summary ===${NC}\n\n"

  printf "  %-15s %s\n" "Backend:" "$BACKEND_TYPE"
  printf "  %-15s %s\n" "Install:" "$INSTALL_MODE"
  printf "  %-15s %s\n" "Listen:" "$LISTEN_ADDR"

  if [[ "$BACKEND_TYPE" == "deepseek" ]]; then
    printf "\n  DeepSeek:\n"
    printf "    %-13s %s\n" "Base URL:" "$DEEPSEEK_BASE_URL"
    printf "    %-13s %s\n" "API Key:" "$(mask_key "$DEEPSEEK_API_KEY")"
  else
    printf "\n  LiteLLM:\n"
    printf "    %-13s %s\n" "Base URL:" "$LITELLM_BASE_URL"
    if [[ -n "$LITELLM_API_KEY" ]]; then
      printf "    %-13s %s\n" "API Key:" "$(mask_key "$LITELLM_API_KEY")"
    else
      printf "    %-13s %s\n" "API Key:" "(no-auth mode)"
    fi
    if [[ ${#MODEL_MAP_KEYS[@]} -gt 0 ]]; then
      printf "    Model Map:\n"
      for i in "${!MODEL_MAP_KEYS[@]}"; do
        printf "      %-28s → %s\n" "${MODEL_MAP_KEYS[$i]}" "${MODEL_MAP_VALS[$i]}"
      done
    fi
  fi

  printf "\n  Claude wrapper:\n"
  printf "    %-15s %s\n" "Default model:" "$ANTHROPIC_MODEL"
  printf "    %-15s %s\n" "Subagent model:" "$CLAUDE_CODE_SUBAGENT_MODEL"
  printf "    %-15s %s\n" "Script:" "./claude-proxy.sh"

  printf "\n  Files to create:\n"
  printf "    config.yaml\n"
  printf "    claude-proxy.sh\n"
  printf "\n"
}

# ---------------------------------------------------------------------------
# Step 7a: Generate config.yaml
# ---------------------------------------------------------------------------

generate_config() {
  local config_path="$SCRIPT_DIR/config.yaml"

  if [[ -f "$config_path" ]]; then
    if ! confirm "config.yaml already exists. Overwrite?" "N"; then
      info "Keeping existing config.yaml"
      return
    fi
  fi

  info "Writing config.yaml..."

  cat > "$config_path" <<EOF
# Generated by setup.sh — $(date -I)
#
# Env var references (\${VAR_NAME}) are interpolated at load time.
# Direct environment variables override YAML values at runtime.

backend:
  type: ${BACKEND_TYPE}

server:
  listen_addr: "${LISTEN_ADDR}"
  client_timeout: "5m"

logging:
  level: "info"
  format: "json"
EOF

  if [[ "$BACKEND_TYPE" == "deepseek" ]]; then
    cat >> "$config_path" <<'EOF'

deepseek:
  api_key: "${DEEPSEEK_API_KEY}"
  api_key_file: ""
EOF
    echo "  base_url: \"${DEEPSEEK_BASE_URL}\"" >> "$config_path"
  else
    cat >> "$config_path" <<'EOF'

litellm:
EOF
    echo "  base_url: \"${LITELLM_BASE_URL}\"" >> "$config_path"
    if [[ -n "$LITELLM_API_KEY" ]]; then
      echo '  api_key: "${LITELLM_API_KEY}"' >> "$config_path"
    else
      echo '  api_key: ""' >> "$config_path"
    fi
    if [[ ${#MODEL_MAP_KEYS[@]} -gt 0 ]]; then
      echo "  model_map:" >> "$config_path"
      for i in "${!MODEL_MAP_KEYS[@]}"; do
        echo "    \"${MODEL_MAP_KEYS[$i]}\": \"${MODEL_MAP_VALS[$i]}\"" >> "$config_path"
      done
    else
      echo "  model_map: {}" >> "$config_path"
    fi
  fi

  ok "config.yaml written"
}

# ---------------------------------------------------------------------------
# Step 7b: Generate claude-proxy.sh
# ---------------------------------------------------------------------------

generate_wrapper() {
  local wrapper_path="$SCRIPT_DIR/claude-proxy.sh"

  if [[ -f "$wrapper_path" ]]; then
    if ! confirm "claude-proxy.sh already exists. Overwrite?" "N"; then
      info "Keeping existing claude-proxy.sh"
      return
    fi
  fi

  info "Writing claude-proxy.sh..."

  cat > "$wrapper_path" <<'WRAPPER_EOF'
#!/usr/bin/env bash
set -euo pipefail

# Claude Code → claude-litellm-proxy
WRAPPER_EOF
  echo "# Generated by setup.sh — $(date -I)" >> "$wrapper_path"
  cat >> "$wrapper_path" <<WRAPPER_EOF

export ANTHROPIC_BASE_URL="\${ANTHROPIC_BASE_URL:-http://${LISTEN_ADDR}}"
export ANTHROPIC_AUTH_TOKEN="\${ANTHROPIC_AUTH_TOKEN:-anything}"

# Default route: ${ANTHROPIC_MODEL}
export ANTHROPIC_MODEL="\${ANTHROPIC_MODEL:-${ANTHROPIC_MODEL}}"

# Pin Claude Code aliases so they hit your proxy keys directly
export ANTHROPIC_DEFAULT_SONNET_MODEL="\${ANTHROPIC_DEFAULT_SONNET_MODEL:-sonnet}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="\${ANTHROPIC_DEFAULT_OPUS_MODEL:-opus}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="\${ANTHROPIC_DEFAULT_HAIKU_MODEL:-haiku}"

# Subagents: ${CLAUDE_CODE_SUBAGENT_MODEL}
export CLAUDE_CODE_SUBAGENT_MODEL="\${CLAUDE_CODE_SUBAGENT_MODEL:-${CLAUDE_CODE_SUBAGENT_MODEL}}"

exec claude "\$@"
WRAPPER_EOF

  chmod +x "$wrapper_path"
  ok "claude-proxy.sh written (chmod +x)"
}

# ---------------------------------------------------------------------------
# Step 7c: Build / Start
# ---------------------------------------------------------------------------

build_and_start() {
  if [[ "$INSTALL_MODE" == "build" ]]; then
    info "Building proxy binary..."
    if (cd "$SCRIPT_DIR" && go build -o claude-go-to-deepseek-proxy .); then
      ok "Build successful: ./claude-go-to-deepseek-proxy"
    else
      err "Build failed. Check the output above for errors."
      exit 1
    fi
  else
    info "Starting proxy with Docker Compose..."
    if (cd "$SCRIPT_DIR" && docker compose up -d); then
      ok "Docker container started"
      info "Waiting for health check..."
      local retries=15
      while [[ $retries -gt 0 ]]; do
        if curl -sf "http://127.0.0.1:8082/health" >/dev/null 2>&1; then
          ok "Proxy is healthy"
          break
        fi
        retries=$((retries - 1))
        sleep 2
      done
      if [[ $retries -eq 0 ]]; then
        warn "Health check timed out. Check status with: docker compose logs"
      fi
    else
      err "Docker Compose failed. Check the output above for errors."
      (cd "$SCRIPT_DIR" && docker compose logs --tail=20 2>/dev/null) || true
      exit 1
    fi
  fi
}

# ---------------------------------------------------------------------------
# Step 7d: PATH setup
# ---------------------------------------------------------------------------

path_setup() {
  if ! has_cmd claude-proxy 2>/dev/null; then
    choose "Add claude-proxy.sh to PATH?" \
      "Yes (symlink to /usr/local/bin/claude-proxy)" \
      "No, I'll run it manually from ./claude-proxy.sh"
    if [[ "$REPLY" == "Yes (symlink to /usr/local/bin/claude-proxy)" ]]; then
      if ln -sf "$SCRIPT_DIR/claude-proxy.sh" /usr/local/bin/claude-proxy 2>/dev/null; then
        ok "Symlink created: /usr/local/bin/claude-proxy"
      elif sudo ln -sf "$SCRIPT_DIR/claude-proxy.sh" /usr/local/bin/claude-proxy; then
        ok "Symlink created (with sudo): /usr/local/bin/claude-proxy"
      else
        warn "Could not create symlink. Add $SCRIPT_DIR to your PATH manually."
      fi
    fi
  else
    ok "claude-proxy already in PATH"
  fi
}

# ---------------------------------------------------------------------------
# Step 7e: Final message
# ---------------------------------------------------------------------------

final_message() {
  printf "\n${GREEN}${BOLD}Setup complete!${NC}\n\n"

  if [[ "$INSTALL_MODE" == "build" ]]; then
    printf "  Start the proxy:\n"
    printf "    ${CYAN}./claude-go-to-deepseek-proxy${NC}\n\n"
  else
    printf "  The proxy is running (docker compose up -d)\n\n"
  fi

  printf "  Run Claude with your proxy:\n"
  printf "    ${CYAN}./claude-proxy.sh${NC}\n\n"

  if has_cmd claude-proxy 2>/dev/null; then
    printf "  Or (symlinked):\n"
    printf "    ${CYAN}claude-proxy${NC}\n\n"
  fi

  printf "  To reconfigure, run ${CYAN}./setup.sh${NC} again.\n\n"
}

generate_env() {
  # Export API key env vars so they're available when the proxy starts
  # (config.yaml uses ${VAR_NAME} references that resolve at runtime)
  if [[ "$BACKEND_TYPE" == "deepseek" ]]; then
    export DEEPSEEK_API_KEY
  else
    export LITELLM_API_KEY
  fi

  if [[ "$INSTALL_MODE" != "docker" ]]; then
    return
  fi

  local env_path="$SCRIPT_DIR/.env"

  if [[ -f "$env_path" ]]; then
    if ! confirm ".env already exists. Overwrite?" "N"; then
      info "Keeping existing .env"
      return
    fi
  fi

  info "Writing .env for Docker Compose..."

  {
    echo "# Generated by setup.sh — $(date -I)"
    echo "BACKEND_TYPE=${BACKEND_TYPE}"
    if [[ "$BACKEND_TYPE" == "deepseek" ]]; then
      echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}"
      echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}"
    else
      echo "LITELLM_BASE_URL=${LITELLM_BASE_URL}"
      [[ -n "$LITELLM_API_KEY" ]] && echo "LITELLM_API_KEY=${LITELLM_API_KEY}"
    fi
    echo "LOG_LEVEL=info"
    echo "LOG_FORMAT=json"
  } > "$env_path"

  ok ".env written"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

printf "\n${BOLD}Claude LiteLLM Proxy — Setup Wizard${NC}\n\n"

check_deps
ask_install_mode
ask_backend_type
ask_backend_config
ask_wrapper_config

show_summary

if ! confirm "Apply this configuration?"; then
  info "Aborted. No files were changed."
  exit 0
fi

generate_config
generate_env
generate_wrapper
build_and_start
path_setup
final_message
