package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"claude-go-to-deepseek-proxy/internal/logger"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestHandler(upstream http.Handler, apiKey string) (*ProxyHandler, *httptest.Server) {
	upstreamServer := httptest.NewServer(upstream)
	h := &ProxyHandler{
		UpstreamURL: upstreamServer.URL,
		APIKey:      apiKey,
		HTTPClient:  &http.Client{Timeout: 30 * time.Second},
		Logger:      logger.Setup("debug", "json"),
	}
	return h, upstreamServer
}

func TestHandler_HealthCheck_NotImplementedInHandler(t *testing.T) {
	// Health check is handled at the router level, not in ProxyHandler
	// ProxyHandler only handles /v1/messages
}

func TestHandler_NormalRequest_TransformsBody(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "test-api-key", r.Header.Get("x-api-key"))

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		messages := req["messages"].([]any)
		msg := messages[0].(map[string]any)
		content := msg["content"].([]any)
		assert.Len(t, content, 1)
		assert.Equal(t, "text", content[0].(map[string]any)["type"])

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "msg_123", "content": []any{}})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-api-key")
	defer upstreamServer.Close()

	requestBody := `{
		"model": "deepseek-v4-pro",
		"max_tokens": 1000,
		"messages": [{
			"role": "assistant",
			"content": [
				{"type": "redacted_thinking", "data": "secret"},
				{"type": "text", "text": "Hello"}
			]
		}]
	}`

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader(requestBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)
	var respBody map[string]any
	json.NewDecoder(resp.Body).Decode(&respBody)
	assert.Equal(t, "msg_123", respBody["id"])
}

func TestHandler_StreamingRequest_Passthrough(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.(http.Flusher).Flush()

		io.WriteString(w, "data: {\"delta\":\"hello\"}\n\n")
		w.(http.Flusher).Flush()
		io.WriteString(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	})

	handler, upstreamServer := newTestHandler(upstream, "test-api-key")
	defer upstreamServer.Close()

	requestBody := `{"model":"deepseek-v4-pro","max_tokens":1000,"stream":true,"messages":[{"role":"user","content":"hi"}]}`

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader(requestBody))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))
	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), "data: {\"delta\":\"hello\"}")
	assert.Contains(t, string(body), "data: [DONE]")
}

func TestHandler_RedactedThinkingRemoved(t *testing.T) {
	var upstreamBody []byte

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "msg_1"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-api-key")
	defer upstreamServer.Close()

	requestBody := `{
		"messages": [{
			"role": "assistant",
			"content": [
				{"type": "redacted_thinking", "data": "secret"},
				{"type": "thinking", "thinking": "visible"},
				{"type": "redacted_thinking", "data": "secret2"},
				{"type": "text", "text": "answer"}
			]
		}]
	}`

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader(requestBody))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var sent map[string]any
	json.Unmarshal(upstreamBody, &sent)
	messages := sent["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	assert.Len(t, content, 2)

	foundThinking := false
	foundText := false
	for _, c := range content {
		cm := c.(map[string]any)
		if cm["type"] == "thinking" {
			foundThinking = true
		}
		if cm["type"] == "text" {
			foundText = true
		}
		assert.NotEqual(t, "redacted_thinking", cm["type"])
	}
	assert.True(t, foundThinking)
	assert.True(t, foundText)
}

func TestHandler_ReasoningContentAlwaysStripped(t *testing.T) {
	var upstreamBody []byte

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "msg_1"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-api-key")
	defer upstreamServer.Close()

	requestBody := `{
		"messages": [{
			"role": "assistant",
			"reasoning_content": "Let me think...",
			"tool_calls": [{"id": "call_1", "type": "function"}],
			"content": "Using tool..."
		}]
	}`

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader(requestBody))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var sent map[string]any
	json.Unmarshal(upstreamBody, &sent)
	messages := sent["messages"].([]any)
	assert.NotContains(t, messages[0].(map[string]any), "reasoning_content")
}

func TestHandler_ReasoningContentStrippedWithoutTools(t *testing.T) {
	var upstreamBody []byte

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "msg_1"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-api-key")
	defer upstreamServer.Close()

	requestBody := `{
		"messages": [{
			"role": "assistant",
			"reasoning_content": "Unneeded thinking",
			"content": "Simple answer"
		}]
	}`

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader(requestBody))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var sent map[string]any
	json.Unmarshal(upstreamBody, &sent)
	messages := sent["messages"].([]any)
	assert.NotContains(t, messages[0].(map[string]any), "reasoning_content")
}

func TestHandler_InvalidJSON_Returns400(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("upstream should not be called")
	})

	handler, upstreamServer := newTestHandler(upstream, "test-api-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader("not json"))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
}

func TestHandler_UpstreamError_Returns502(t *testing.T) {
	handler, upstreamServer := newTestHandler(nil, "test-api-key")
	upstreamServer.Close() // simulate upstream being down

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusBadGateway, resp.StatusCode)
}

func TestHandler_AuthorizationHeaderReplaced(t *testing.T) {
	var upstreamHeaders http.Header

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHeaders = r.Header
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "msg_1"})
	})

	handler, upstreamServer := newTestHandler(upstream, "deepseek-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	req, _ := http.NewRequest("POST", proxyServer.URL+"/v1/messages",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer claude-key")
	req.Header.Set("X-Request-Id", "test-req-123")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, "deepseek-key", upstreamHeaders.Get("x-api-key"))
	assert.Empty(t, upstreamHeaders.Get("Authorization"))
}

func TestHandler_RequestIDFromHeader(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	req, _ := http.NewRequest("POST", proxyServer.URL+"/v1/messages",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", "my-custom-id")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestHandler_UpstreamNon200_PropagatesStatus(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]any{"error": "rate limited"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)

	body, _ := io.ReadAll(resp.Body)
	assert.Contains(t, string(body), "rate limited")
}

func TestHandler_EmptyBody_Propagates(t *testing.T) {
	var upstreamBody []byte

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "empty_ok"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json",
		strings.NewReader(`{}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var sent map[string]any
	json.Unmarshal(upstreamBody, &sent)
	assert.NotContains(t, sent, "messages")
}

func TestHandler_NonArrayMessages_Propagates(t *testing.T) {
	var upstreamBody []byte

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"messages":"not_an_array"}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var sent map[string]any
	json.Unmarshal(upstreamBody, &sent)
	assert.Equal(t, "not_an_array", sent["messages"])
}

func TestHandler_NonMapMessageElements_Propagates(t *testing.T) {
	var upstreamBody []byte

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	})

	handler, upstreamServer := newTestHandler(upstream, "test-key")
	defer upstreamServer.Close()

	proxyServer := httptest.NewServer(handler)
	defer proxyServer.Close()

	body := `{"messages":["not_a_map",{"role":"assistant","content":[{"type":"redacted_thinking","data":"x"},{"type":"text","text":"hi"}]}]}`
	resp, err := http.Post(proxyServer.URL+"/v1/messages", "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var sent map[string]any
	json.Unmarshal(upstreamBody, &sent)
	messages := sent["messages"].([]any)
	assert.Len(t, messages, 2)
	msg := messages[1].(map[string]any)
	contentArr := msg["content"].([]any)
	assert.Len(t, contentArr, 1)
	assert.Equal(t, "text", contentArr[0].(map[string]any)["type"])
}
