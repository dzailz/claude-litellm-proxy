package proxy

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

type ProxyHandler struct {
	UpstreamURL string
	APIKey      string
	HTTPClient  *http.Client
	Logger      *slog.Logger
}

func generateRequestID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func maskAPIKey(key string) string {
	if len(key) <= 8 {
		return strings.Repeat("*", len(key))
	}
	return key[:4] + "..." + key[len(key)-4:]
}

func (h *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	reqID := r.Header.Get("X-Request-Id")
	if reqID == "" {
		reqID = generateRequestID()
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		h.logError(reqID, r, 400, start, "failed to read request body", err)
		http.Error(w, `{"error":"failed to read request body"}`, http.StatusBadRequest)
		return
	}
	r.Body.Close()

	var body map[string]any
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		h.logError(reqID, r, 400, start, "invalid JSON body", err)
		http.Error(w, `{"error":"invalid JSON body"}`, http.StatusBadRequest)
		return
	}

	removedBlocks := 0
	hasToolCalls := false
	reasoningStripped := 0

	messagesRaw, exists := body["messages"]
	if exists {
		if msgArr, ok := messagesRaw.([]any); ok {
			messages := make([]map[string]any, len(msgArr))
			for i, m := range msgArr {
				if mm, ok := m.(map[string]any); ok {
					messages[i] = mm
				}
			}

			messages, removedBlocks = RemoveRedactedThinking(messages)
			body["messages"] = messages

			hasToolCalls = ShouldKeepReasoningContent(messages)
			if !hasToolCalls {
				reasoningStripped = StripReasoningContent(messages)
			}
		}
	}

	modifiedBody, err := json.Marshal(body)
	if err != nil {
		h.logError(reqID, r, 500, start, "failed to re-serialize body", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}

	h.Logger.Debug("request transformed",
		"request_id", reqID,
		"removed_blocks", removedBlocks,
		"has_tool_calls", hasToolCalls,
		"reasoning_stripped", reasoningStripped,
	)

	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		h.UpstreamURL+r.URL.RequestURI(), bytes.NewReader(modifiedBody))
	if err != nil {
		h.logError(reqID, r, 500, start, "failed to create upstream request", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}

	upstreamReq.Header.Set("Content-Type", "application/json")
	upstreamReq.Header.Set("Host", "api.deepseek.com")
	upstreamReq.Header.Set("x-api-key", h.APIKey)

	for key, values := range r.Header {
		lowerKey := strings.ToLower(key)
		if lowerKey == "x-api-key" || lowerKey == "authorization" ||
			lowerKey == "host" || lowerKey == "content-length" {
			continue
		}
		for _, v := range values {
			upstreamReq.Header.Add(key, v)
		}
	}

	resp, err := h.HTTPClient.Do(upstreamReq)
	if err != nil {
		statusCode := http.StatusBadGateway
		if strings.Contains(err.Error(), "timeout") ||
			strings.Contains(err.Error(), "deadline exceeded") {
			statusCode = http.StatusGatewayTimeout
		}
		h.logError(reqID, r, statusCode, start, "upstream request failed", err)
		http.Error(w, `{"error":"upstream request failed"}`, statusCode)
		return
	}
	defer resp.Body.Close()

	isStream := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")

	if isStream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(resp.StatusCode)

		flusher, ok := w.(http.Flusher)
		if !ok {
			h.logError(reqID, r, 500, start, "streaming not supported", nil)
			return
		}
		flusher.Flush()

		if err := ProcessStream(resp.Body, w); err != nil {
			h.Logger.Warn("stream error",
				"request_id", reqID,
				"error", err,
			)
		}

		h.logRequest(reqID, r, resp.StatusCode, start, removedBlocks, hasToolCalls)
		return
	}

	for key, values := range resp.Header {
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)

	h.logRequest(reqID, r, resp.StatusCode, start, removedBlocks, hasToolCalls)
}

func (h *ProxyHandler) logRequest(reqID string, r *http.Request, status int, start time.Time, removedBlocks int, hasToolCalls bool) {
	h.Logger.Info("request completed",
		"request_id", reqID,
		"method", r.Method,
		"url", r.URL.String(),
		"status_code", status,
		"duration", time.Since(start).String(),
		"thinking_blocks_removed", removedBlocks,
		"has_tool_calls", hasToolCalls,
		"api_key_preview", maskAPIKey(h.APIKey),
	)
}

func (h *ProxyHandler) logError(reqID string, r *http.Request, status int, start time.Time, msg string, err error) {
	attrs := []any{
		"request_id", reqID,
		"method", r.Method,
		"url", r.URL.String(),
		"status_code", status,
		"duration", time.Since(start).String(),
		"error_message", msg,
	}
	if err != nil {
		attrs = append(attrs, "error", err.Error())
	}
	h.Logger.Error(msg, attrs...)
}
