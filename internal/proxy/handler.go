package proxy

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"claude-go-to-deepseek-proxy/internal/backend"
	"claude-go-to-deepseek-proxy/internal/compaction"
)

// ProxyHandler is the HTTP handler that receives Anthropic-format requests
// from clients and proxies them to an upstream backend. The Backend interface
// abstracts away the upstream API details (DeepSeek, LiteLLM, etc.) so the
// handler stays generic.
type ProxyHandler struct {
	Backend    backend.Backend
	HTTPClient *http.Client
	Logger     *slog.Logger
	// Compactor performs transparent conversation compaction when the
	// estimated token count exceeds the configured threshold. May be nil
	// if compaction is disabled.
	Compactor *compaction.Compactor
}

func generateRequestID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func (h *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	reqID := r.Header.Get("X-Request-Id")
	if reqID == "" {
		reqID = generateRequestID()
	}

	// Read and parse the incoming JSON body.
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

	// Apply conversation compaction if enabled and the request exceeds
	// the token threshold. Compaction summarizes older messages to fit
	// within the upstream model's context window. On failure, we log
	// the error and proceed with the original body (graceful degradation).
	if h.Compactor != nil && h.Compactor.ShouldCompact(body) {
		compacted, err := h.Compactor.Compact(r.Context(), body)
		if err != nil {
			h.Logger.Warn("compaction failed, using original body",
				"request_id", reqID,
				"error", err,
			)
		} else {
			body = compacted
		}
	}

	// Delegate request preparation to the backend. The backend handles
	// message transformation, header injection, and URL construction.
	upstreamReq, err := h.Backend.PrepareRequest(r.Context(), body, r.Header)
	if err != nil {
		h.logError(reqID, r, 500, start, "failed to prepare upstream request", err)
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
		return
	}

	// Execute the upstream request.
	resp, err := h.HTTPClient.Do(upstreamReq)
	if err != nil {
		statusCode := http.StatusBadGateway
		if strings.Contains(err.Error(), "timeout") ||
			strings.Contains(err.Error(), "deadline exceeded") ||
			strings.Contains(err.Error(), "Timeout") {
			statusCode = http.StatusGatewayTimeout
		}
		h.logError(reqID, r, statusCode, start, "upstream request failed", err)
		http.Error(w, `{"error":"upstream request failed"}`, statusCode)
		return
	}
	// NOTE: The response body is owned by the backend's Handle methods.
	// Do NOT defer resp.Body.Close() here.

	isStream := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")

	if isStream && h.Backend.NeedsStreamTransform() {
		// Streaming with format translation (e.g., OpenAI SSE → Anthropic SSE).
		// Set SSE headers before delegating to the backend so the client
		// gets proper streaming headers immediately.
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(resp.StatusCode)

		flusher, ok := w.(http.Flusher)
		if !ok {
			resp.Body.Close()
			h.logError(reqID, r, 500, start, "streaming not supported", nil)
			return
		}
		flusher.Flush()

		err = h.Backend.HandleStream(resp, w, h.Logger)
	} else {
		// Non-streaming or passthrough streaming (the backend's native
		// protocol matches what the client expects — no transformation needed).
		err = h.Backend.HandleResponse(resp, w, h.Logger)
	}

	if err != nil {
		h.Logger.Warn("response handling error",
			"request_id", reqID,
			"error", err,
		)
	}

	h.logRequest(reqID, r, upstreamReq, resp.StatusCode, start)
}

func (h *ProxyHandler) logRequest(reqID string, r *http.Request, upstreamReq *http.Request, status int, start time.Time) {
	h.Logger.Info("request completed",
		"request_id", reqID,
		"method", r.Method,
		"url", r.URL.String(),
		"upstream_url", upstreamReq.URL.String(),
		"status_code", status,
		"duration", time.Since(start).String(),
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
