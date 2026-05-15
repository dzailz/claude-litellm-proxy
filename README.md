# claude-go-to-deepseek-proxy

HTTP-прокси-сервер на Go, обеспечивающий совместимость Claude Code с API DeepSeek
(модели V4 Pro, V4 Flash).

## Проблема

Claude Code при использовании DeepSeek API выдаёт ошибку:

```
400 {"error":{"message":"The `content[].thinking` in the thinking mode must be passed back to the API."}}
```

**Причины:**
- DeepSeek API **не поддерживает** `redacted_thinking` блоки в Anthropic-формате
- `reasoning_content` требует особой обработки: передаётся обратно **только** при наличии tool calls
- Формат thinking-блоков Claude Code (content-массив) и DeepSeek API (отдельное поле `reasoning_content`) различается

## Как это работает

Прокси нормализует запросы от Claude Code перед отправкой в DeepSeek:

### DeepSeek Mode (direct Anthropic-compatible API)

```
Claude Code                Proxy (127.0.0.1:8082)          DeepSeek API
    │                           │                               │
    │ POST /v1/messages         │                               │
    │ (Anthropic format)        │                               │
    ├──────────────────────────►│                               │
    │                           │ 1. Parse & normalize body     │
    │                           │ 2. Remove redacted_thinking   │
    │                           │ 3. Manage reasoning_content   │
    │                           │                               │
    │                           │ POST /anthropic/v1/messages   │
    │                           ├──────────────────────────────►│
    │                           │                               │
    │                           │ ◄─────────────────────────────┤
    │                           │   (SSE stream or JSON)        │
    │ ◄─────────────────────────┤                               │
    │   (passthrough)           │                               │
```

### LiteLLM Mode (OpenAI bridge for corporate proxies)

```
Claude Code                Proxy (127.0.0.1:8082)          LiteLLM Server
    │                           │                               │
    │ POST /v1/messages         │                               │
    │ (Anthropic format)        │                               │
    ├──────────────────────────►│                               │
    │                           │ 1. Anthropic → OpenAI format  │
    │                           │ 2. Map model names            │
    │                           │                               │
    │                           │ POST /v1/chat/completions     │
    │                           ├──────────────────────────────►│
    │                           │                               │
    │                           │ ◄─────────────────────────────┤
    │                           │   (OpenAI SSE or JSON)        │
    │                           │ 3. OpenAI → Anthropic format  │
    │ ◄─────────────────────────┤                               │
    │   (translated)            │                               │
```

1. **Удаляет `redacted_thinking` блоки** — неподдерживаемый тип контента (DeepSeek mode)
2. **Управляет `reasoning_content`** — сохраняет только при наличии tool calls в истории (DeepSeek mode)
3. **Полная трансляция форматов** — Anthropic ↔ OpenAI через LiteLLM (LiteLLM mode)
4. **Прозрачно проксирует** все остальные части запроса/ответа
5. **Поддерживает стриминг** (SSE, `text/event-stream`) в обоих режимах

## Configuration File

The proxy supports YAML-based configuration via `config.yaml`. When a config file is provided, all settings can be declared declaratively. Environment variables still take precedence for overrides.

To use a config file:

```bash
# Copy the example and customize
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
export CONFIG_FILE=./config.yaml
./claude-go-to-deepseek-proxy
```

### config.yaml format

```yaml
# Backend selection: "deepseek" or "litellm"
backend:
  type: deepseek

# HTTP server settings
server:
  listen_addr: "127.0.0.1:8082"
  client_timeout: "120s"

# Logging
logging:
  level: "info"
  format: "json"

# DeepSeek backend
deepseek:
  api_key: "${DEEPSEEK_API_KEY}"
  api_key_file: ""
  base_url: "https://api.deepseek.com/anthropic"

# LiteLLM backend
litellm:
  base_url: "http://localhost:4000"
  api_key: "${LITELLM_API_KEY}"
  model_map:
    "deepseek-v4-pro": "deepseek/deepseek-chat"
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514"
```

The `${VAR_NAME}` syntax interpolates environment variables at load time. When no config file is available, zero-config defaults are used (DeepSeek mode, port 8082).

## Быстрый старт

### Setup Wizard (recommended)

The interactive setup wizard generates `config.yaml` and `claude-proxy.sh` for you:

```bash
./setup.sh
```

The wizard will:
1. Check for required dependencies (Go, Docker)
2. Ask how to run the proxy (build locally or Docker Compose)
3. Select backend (DeepSeek or LiteLLM) and configure it
4. Set up model mapping (LiteLLM mode)
5. Configure Claude Code wrapper settings
6. Generate all config files and optionally build/start the proxy

After setup, run Claude through the wrapper:

```bash
./claude-proxy.sh
```

### Manual Setup

### Переменные окружения

| Переменная              | По умолчанию            | Описание                                                                                           |
|-------------------------|-------------------------|----------------------------------------------------------------------------------------------------|
| `LISTEN_ADDR`           | `127.0.0.1:8082`        | Адрес прослушивания                                                                                |
| `DEEPSEEK_API_KEY`      | —                       | API-ключ DeepSeek                                                                                  |
| `DEEPSEEK_API_KEY_FILE` | —                       | Путь к файлу с ключом (приоритетнее)                                                               |
| `LOG_LEVEL`             | `info`                  | Уровень: `debug`, `info`, `warn`, `error`                                                          |
| `LOG_FORMAT`            | `json`                  | Формат логов: `json`, `text`                                                                       |
| `CLIENT_TIMEOUT`        | `120s`                  | Таймаут подключения (dial, TLS, response headers). **Чтение тела ответа (stream) — без таймаута.** |
| `BACKEND_TYPE`          | `deepseek`              | Режим: `deepseek` или `litellm`                                                                    |
| `LITELLM_API_KEY`       | —                       | Виртуальный ключ LiteLLM (опционально)                                                             |
| `LITELLM_BASE_URL`      | `http://localhost:4000` | URL LiteLLM-инстанса                                                                               |
| `CONFIG_FILE`           | —                       | Путь к YAML-файлу конфигурации (`config.yaml`)                                                     |

### Из исходников

```bash
go build -o claude-go-to-deepseek-proxy .

# DeepSeek mode (default)
export DEEPSEEK_API_KEY="sk-your-key-here"
./claude-go-to-deepseek-proxy

# Or with config file
cp config.example.yaml config.yaml
export CONFIG_FILE=./config.yaml
./claude-go-to-deepseek-proxy
```

### Docker

```bash
docker build -t claude-deepseek-proxy .
docker run -d --name deepseek-proxy \
  -p 127.0.0.1:8082:8082 \
  -e DEEPSEEK_API_KEY="sk-your-key-here" \
  claude-deepseek-proxy
```

### Docker Compose

```bash
# .env
DEEPSEEK_API_KEY=sk-your-key-here
```

```bash
docker compose up -d
docker compose logs -f
```

### Конфигурация Claude Code

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8082"
export ANTHROPIC_AUTH_TOKEN="anything"
export ANTHROPIC_MODEL="deepseek-v4-pro"
export CLAUDE_CODE_EFFORT_LEVEL="max"
```

> **Важно:** `ANTHROPIC_BASE_URL` указывает только на хост и порт, **без пути**.
> Anthropic SDK сам добавляет `/v1/messages?beta=true` к base_url.

## LiteLLM Support

LiteLLM mode routes requests through a LiteLLM proxy server instead of directly to DeepSeek. This is useful for:

- **Corporate environments** — route all AI traffic through a central LiteLLM gateway for cost tracking, rate limiting, and access control
- **Multiple model backends** — use models from OpenAI, Anthropic, DeepSeek, and others through a single endpoint
- **Custom model mapping** — map Claude model names to any LiteLLM-supported model identifier

### Configuration

LiteLLM mode requires a LiteLLM instance. You can self-host LiteLLM or use a managed instance.

**Option 1: Environment variables**

```bash
export BACKEND_TYPE=litellm
export LITELLM_BASE_URL="http://litellm-host:4000"
export LITELLM_API_KEY="sk-your-liteLLM-virtual-key"  # optional
./claude-go-to-deepseek-proxy
```

**Option 2: config.yaml**

```yaml
backend:
  type: litellm

litellm:
  base_url: "http://localhost:4000"
  api_key: "${LITELLM_API_KEY}"
  model_map:
    "deepseek-v4-pro": "deepseek/deepseek-chat"
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514"
```

```bash
export CONFIG_FILE=./config.yaml
./claude-go-to-deepseek-proxy
```

### Docker Compose with LiteLLM

```bash
# .env
BACKEND_TYPE=litellm
LITELLM_BASE_URL=http://litellm-host:4000
LITELLM_API_KEY=sk-your-virtual-key
```

```bash
docker compose up -d
```

### How Claude Code Connects

Claude Code connects to the proxy **exactly the same way** regardless of backend mode:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8082"
export ANTHROPIC_AUTH_TOKEN="anything"
export ANTHROPIC_MODEL="deepseek-v4-pro"
```

> The proxy handles all format translation transparently. Claude Code always speaks Anthropic format — the proxy translates to OpenAI format when in LiteLLM mode, and passes through directly in DeepSeek mode.

### Model Mapping

The `model_map` in the LiteLLM config translates Claude-side model names to LiteLLM model identifiers:

| Claude Model (ANTHROPIC_MODEL) | LiteLLM Model Identifier             |
|--------------------------------|--------------------------------------|
| `deepseek-v4-pro`              | `deepseek/deepseek-chat`             |
| `claude-sonnet-4-20250514`     | `anthropic/claude-sonnet-4-20250514` |

Unmapped models are passed through as-is.

## Эндпоинты

| Метод  | Путь                     | Описание                                        |
|--------|--------------------------|-------------------------------------------------|
| `POST` | `/v1/messages`           | Проксирование Messages API                      |
| `POST` | `/v1/messages?beta=true` | Аналогично (Claude Code передаёт `?beta=true`)  |
| `GET`  | `/health`                | Health check — возвращает `200 {"status":"ok"}` |

## Логирование

Каждый запрос логируется в формате JSON (или `text` при `LOG_FORMAT=text`):

```json
{
  "time": "2026-05-01T17:42:46+03:00",
  "level": "INFO",
  "msg": "request completed",
  "request_id": "e42adff5-9f1b-31a3-cfdb-d1f0e137ece3",
  "method": "POST",
  "url": "/v1/messages",
  "upstream_url": "https://api.deepseek.com/anthropic/v1/messages",
  "status_code": 200,
  "duration": "589.9µs"
}
```

API-ключ не попадает в логи ни в каком виде.

## Структура проекта

```
claude-go-to-deepseek-proxy/
├── main.go
├── go.mod / go.sum
├── config.example.yaml          # Example YAML config (copy to config.yaml)
├── internal/
│   ├── backend/
│   │   ├── backend.go           # Backend interface definition
│   │   ├── deepseek.go          # DeepSeek backend (passthrough + transform)
│   │   └── litellm.go           # LiteLLM backend (Anthropic ↔ OpenAI)
│   ├── config/
│   │   └── config.go            # YAML config loading with env var overrides
│   ├── proxy/
│   │   ├── handler.go           # HTTP handler: parse → transform → forward
│   │   ├── handler_test.go      # Integration tests
│   │   ├── messages.go          # Message transformation
│   │   ├── messages_test.go     # Unit tests
│   │   ├── stream.go            # SSE stream passthrough
│   │   └── stream_test.go       # Unit tests
│   └── logger/
│       └── logger.go            # slog setup
├── Dockerfile
├── docker-compose.yaml
├── setup.sh                      # Interactive setup wizard
├── claude-proxy.sh               # Generated wrapper (by setup.sh)
└── README.md
```

## Тестирование

```bash
go test ./internal/... -v -count=1

# Покрытие
go test ./internal/... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

## Troubleshooting

### Прокси не запускается

```bash
# Проверьте, что ключ задан
echo $DEEPSEEK_API_KEY

# Или через файл
echo "sk-..." > /tmp/deepseek.key
export DEEPSEEK_API_KEY_FILE=/tmp/deepseek.key
```

### Claude Code выдаёт 400 с thinking

Убедитесь, что прокси запущен и `ANTHROPIC_BASE_URL` указывает на него:

```bash
curl -s http://127.0.0.1:8082/health
# {"status":"ok"}
```

### Логи прокси

```bash
# Native
LOG_LEVEL=debug ./claude-go-to-deepseek-proxy

# Docker
docker logs -f claude-deepseek-proxy

# Docker Compose
docker compose logs -f
```
