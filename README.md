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

1. **Удаляет `redacted_thinking` блоки** — неподдерживаемый тип контента
2. **Управляет `reasoning_content`** — сохраняет только при наличии tool calls в истории
3. **Прозрачно проксирует** все остальные части запроса/ответа
4. **Поддерживает стриминг** (SSE, `text/event-stream`)

## Быстрый старт

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LISTEN_ADDR` | `127.0.0.1:8082` | Адрес прослушивания |
| `DEEPSEEK_API_KEY` | — | API-ключ DeepSeek |
| `DEEPSEEK_API_KEY_FILE` | — | Путь к файлу с ключом (приоритетнее) |
| `LOG_LEVEL` | `info` | Уровень: `debug`, `info`, `warn`, `error` |
| `LOG_FORMAT` | `json` | Формат логов: `json`, `text` |
| `CLIENT_TIMEOUT` | `120s` | Таймаут запросов к DeepSeek API |

### Из исходников

```bash
go build -o claude-go-to-deepseek-proxy .
export DEEPSEEK_API_KEY="sk-your-key-here"
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

## Эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/v1/messages` | Проксирование Messages API |
| `POST` | `/v1/messages?beta=true` | Аналогично (Claude Code передаёт `?beta=true`) |
| `GET` | `/health` | Health check — возвращает `200 {"status":"ok"}` |

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
  "status_code": 200,
  "duration": "589.9µs",
  "thinking_blocks_removed": 1,
  "has_tool_calls": false,
  "api_key_preview": "sk-a...b12c"
}
```

API-ключ логируется только в виде первых и последних 4 символов.

## Структура проекта

```
claude-go-to-deepseek-proxy/
├── main.go
├── go.mod / go.sum
├── internal/
│   ├── proxy/
│   │   ├── handler.go         # HTTP handler: parse → transform → forward
│   │   ├── handler_test.go    # Integration tests
│   │   ├── messages.go        # Message transformation
│   │   ├── messages_test.go   # Unit tests
│   │   ├── stream.go          # SSE stream passthrough
│   │   └── stream_test.go     # Unit tests
│   └── logger/
│       └── logger.go          # slog setup
├── Dockerfile
├── docker-compose.yaml
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
