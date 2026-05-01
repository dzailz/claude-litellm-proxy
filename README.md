# claude-go-to-deepseek-proxy

HTTP-прокси-сервер на Go, обеспечивающий совместимость Claude Code с API DeepSeek (модели V4 Pro, V4 Flash).

## Проблема

Claude Code при использовании DeepSeek API выдаёт ошибку:

```
400 {"error":{"message":"The `content[].thinking` in the thinking mode must be passed back to the API."}}
```

**Причины:**
- DeepSeek API не поддерживает `redacted_thinking` блоки в Anthropic-формате
- `reasoning_content` требует особой обработки: передаётся обратно только при наличии tool calls
- Формат thinking-блоков Claude Code и DeepSeek API различается

## Как это работает

Прокси нормализует запросы от Claude Code перед отправкой в DeepSeek:

1. **Удаляет `redacted_thinking` блоки** — неподдерживаемый тип контента
2. **Управляет `reasoning_content`** — сохраняет только при наличии tool calls в истории
3. **Прозрачно проксирует** все остальные части запроса/ответа
4. **Поддерживает стриминг** (SSE) ответы

## Быстрый старт

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LISTEN_ADDR` | `127.0.0.1:8082` | Адрес прослушивания |
| `DEEPSEEK_API_KEY` | — | API-ключ DeepSeek (обязательно, если не указан файл) |
| `DEEPSEEK_API_KEY_FILE` | — | Путь к файлу с API-ключом |
| `LOG_LEVEL` | `info` | Уровень логирования (debug, info, warn, error) |
| `LOG_FORMAT` | `json` | Формат логов (json, text) |
| `CLIENT_TIMEOUT` | `120s` | Таймаут запросов к DeepSeek API |

### Сборка и запуск

```bash
go build -o claude-go-to-deepseek-proxy .
export DEEPSEEK_API_KEY="sk-your-key-here"
./claude-go-to-deepseek-proxy
```

### Docker

```bash
docker build -t claude-deepseek-proxy .
docker run -p 8082:8082 \
  -e DEEPSEEK_API_KEY="sk-your-key-here" \
  claude-deepseek-proxy
```

### Конфигурация Claude Code

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8082/v1/messages?beta=true"
export ANTHROPIC_AUTH_TOKEN="anything"
export ANTHROPIC_MODEL="deepseek-v4-pro"
export CLAUDE_CODE_EFFORT_LEVEL="max"
```

## Доступные эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/v1/messages` | Проксирование Messages API (включая `?beta=true`) |
| `GET` | `/health` | Проверка работоспособности (200 OK) |

## Логирование

Каждый запрос логируется в формате JSON:

```json
{
  "request_id": "uuid",
  "method": "POST",
  "url": "/v1/messages",
  "status_code": 200,
  "duration": "1.2s",
  "thinking_blocks_removed": 3,
  "has_tool_calls": true,
  "api_key_preview": "sk-a...b12c"
}
```

API-ключ логируется только в виде первых и последних 4 символов.

## Поддерживаемые модели

Прокси не привязан к конкретной модели. Все доступные модели DeepSeek работают:
- `deepseek-v4-pro`
- `deepseek-v4-flash`
- `deepseek-reasoner`
- и другие

## Структура проекта

```
claude-go-to-deepseek-proxy/
├── main.go
├── go.mod
├── go.sum
├── internal/
│   ├── proxy/
│   │   ├── handler.go
│   │   ├── handler_test.go
│   │   ├── messages.go
│   │   ├── messages_test.go
│   │   ├── stream.go
│   │   └── stream_test.go
│   └── logger/
│       └── logger.go
├── Dockerfile
└── README.md
```

## Тестирование

```bash
go test ./internal/... -v -count=1
```
