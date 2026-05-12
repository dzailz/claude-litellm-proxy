// Package backend integration tests for the DeepSeekBackend (passthrough).
//
// Tests verify end-to-end behavior using mock HTTP servers (httptest),
// following the same patterns as proxy/handler_test.go:
//   - httptest.NewServer with http.HandlerFunc for upstream mocking
//   - testify/assert for soft assertions, testify/require for fatal
//   - AAA (Arrange-Act-Assert) structure throughout
package backend

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testLogger returns a discard logger shared across all backend tests.
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// ============================================================================
// DeepSeekBackend — PrepareRequest Tests
// ============================================================================

// TestDeepSeekBackend_PrepareRequest_SendsAPIKey verifies that the x-api-key
// header is set correctly on the outgoing request.
func TestDeepSeekBackend_PrepareRequest_SendsAPIKey(t *testing.T) {
	// Arrange
	var receivedKey string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedKey = r.Header.Get("x-api-key")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test-key-deepseek",
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), map[string]any{"messages": []any{}}, http.Header{})
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "sk-test-key-deepseek", receivedKey)
}

// TestDeepSeekBackend_PrepareRequest_SetsHostHeader verifies the Host header
// is set to api.deepseek.com as required by DeepSeek's API.
func TestDeepSeekBackend_PrepareRequest_SetsHostHeader(t *testing.T) {
	// Arrange
	var receivedHost string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHost = r.Host
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test",
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), map[string]any{"messages": []any{}}, http.Header{})
	require.NoError(t, err)

	// Assert — check req.Host field on the prepared request
	assert.Equal(t, "api.deepseek.com", req.Host)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert — server should also receive the correct Host via r.Host
	assert.Equal(t, "api.deepseek.com", receivedHost)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

// TestDeepSeekBackend_PrepareRequest_RemovesRedactedThinking verifies that
// content blocks of type "redacted_thinking" are removed from the request
// body before forwarding to DeepSeek's API.
func TestDeepSeekBackend_PrepareRequest_RemovesRedactedThinking(t *testing.T) {
	// Arrange
	var upstreamBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &upstreamBody)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test",
	}

	anthropicBody := map[string]any{
		"model": "deepseek-v4-pro",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "redacted_thinking", "data": "secret"},
					map[string]any{"type": "thinking", "thinking": "visible"},
					map[string]any{"type": "redacted_thinking", "data": "secret2"},
					map[string]any{"type": "text", "text": "answer"},
				},
			},
		},
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), anthropicBody, http.Header{})
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	messages := upstreamBody["messages"].([]any)
	require.Len(t, messages, 1)
	msg := messages[0].(map[string]any)
	content := msg["content"].([]any)
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
		assert.NotEqual(t, "redacted_thinking", cm["type"],
			"redacted_thinking should have been removed")
	}
	assert.True(t, foundThinking, "thinking block should be preserved")
	assert.True(t, foundText, "text block should be preserved")
}

// TestDeepSeekBackend_PrepareRequest_StripsReasoningContent verifies that
// the reasoning_content field is removed from messages before forwarding.
func TestDeepSeekBackend_PrepareRequest_StripsReasoningContent(t *testing.T) {
	// Arrange
	var upstreamBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &upstreamBody)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test",
	}

	anthropicBody := map[string]any{
		"messages": []any{
			map[string]any{
				"role":              "assistant",
				"reasoning_content": "Let me think...",
				"content":           "Answer",
			},
		},
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), anthropicBody, http.Header{})
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert
	messages := upstreamBody["messages"].([]any)
	msg := messages[0].(map[string]any)
	_, hasReasoning := msg["reasoning_content"]
	assert.False(t, hasReasoning, "reasoning_content should be stripped")
}

// TestDeepSeekBackend_PrepareRequest_StripsServerToolUse verifies that
// server_tool_use and other unsupported content block types are removed.
func TestDeepSeekBackend_PrepareRequest_StripsServerToolUse(t *testing.T) {
	// Arrange
	var upstreamBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &upstreamBody)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test",
	}

	anthropicBody := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "server_tool_use", "id": "stu_1"},
					map[string]any{"type": "text", "text": "visible"},
					map[string]any{"type": "web_search_tool_result", "data": "x"},
					map[string]any{"type": "image", "source": map[string]any{"data": "abc"}},
					map[string]any{"type": "text", "text": "more"},
				},
			},
		},
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), anthropicBody, http.Header{})
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert — only text blocks should remain (image also unsupported)
	messages := upstreamBody["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	assert.Len(t, content, 2)
	for _, c := range content {
		cm := c.(map[string]any)
		assert.Equal(t, "text", cm["type"])
	}
}

// TestDeepSeekBackend_PrepareRequest_NonArrayMessagesPropagates verifies that
// non-array messages pass through without transformation attempts.
func TestDeepSeekBackend_PrepareRequest_NonArrayMessagesPropagates(t *testing.T) {
	// Arrange
	var upstreamBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &upstreamBody)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test",
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), map[string]any{"messages": "string_not_array"}, http.Header{})
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert
	assert.Equal(t, "string_not_array", upstreamBody["messages"])
}

// TestDeepSeekBackend_PrepareRequest_UserMessagePreserved verifies that
// user-role messages are not modified (only assistant messages are filtered).
func TestDeepSeekBackend_PrepareRequest_UserMessagePreserved(t *testing.T) {
	// Arrange
	var upstreamBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &upstreamBody)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "ok"})
	}))
	defer upstream.Close()

	backend := &DeepSeekBackend{
		BaseURL: upstream.URL,
		APIKey:  "sk-test",
	}

	anthropicBody := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "redacted_thinking", "data": "should-survive"},
					map[string]any{"type": "text", "text": "hello"},
				},
			},
		},
	}

	// Act
	req, err := backend.PrepareRequest(context.Background(), anthropicBody, http.Header{})
	require.NoError(t, err)
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	// Assert — user messages are not filtered, redacted_thinking survives
	messages := upstreamBody["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	assert.Len(t, content, 2)
	types := []string{}
	for _, c := range content {
		types = append(types, c.(map[string]any)["type"].(string))
	}
	assert.Contains(t, types, "redacted_thinking")
	assert.Contains(t, types, "text")
}

// ============================================================================
// DeepSeekBackend — HandleResponse / HandleStream Tests
// ============================================================================

// TestDeepSeekBackend_HandleResponse_Passthrough verifies that the response
// body, status code, and headers are forwarded without modification.
func TestDeepSeekBackend_HandleResponse_Passthrough(t *testing.T) {
	// Arrange
	upstreamBody := map[string]any{
		"id":      "msg_test123",
		"type":    "message",
		"role":    "assistant",
		"model":   "deepseek-v4-pro",
		"content": []any{map[string]any{"type": "text", "text": "Hello from DeepSeek"}},
	}
	bodyBytes, err := json.Marshal(upstreamBody)
	require.NoError(t, err)

	mockResp := &http.Response{
		StatusCode: http.StatusOK,
		Header: http.Header{
			"Content-Type":    {"application/json"},
			"X-Custom-Header": {"test-value"},
		},
		Body: io.NopCloser(strings.NewReader(string(bodyBytes))),
	}

	backend := &DeepSeekBackend{}
	rec := httptest.NewRecorder()

	// Act
	err = backend.HandleResponse(mockResp, rec, testLogger())
	require.NoError(t, err)

	// Assert
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, "application/json", rec.Header().Get("Content-Type"))
	assert.Equal(t, "test-value", rec.Header().Get("X-Custom-Header"))

	var parsedBody map[string]any
	err = json.Unmarshal(rec.Body.Bytes(), &parsedBody)
	require.NoError(t, err)
	assert.Equal(t, "msg_test123", parsedBody["id"])
	assert.Equal(t, "Hello from DeepSeek",
		parsedBody["content"].([]any)[0].(map[string]any)["text"])
}

// TestDeepSeekBackend_HandleResponse_PropagatesErrorStatus verifies that
// non-200 status codes and error bodies are forwarded to the client.
func TestDeepSeekBackend_HandleResponse_PropagatesErrorStatus(t *testing.T) {
	// Arrange
	errorBody := `{"error":{"type":"rate_limit","message":"Too many requests"}}`
	mockResp := &http.Response{
		StatusCode: http.StatusTooManyRequests,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(errorBody)),
	}

	backend := &DeepSeekBackend{}
	rec := httptest.NewRecorder()

	// Act
	err := backend.HandleResponse(mockResp, rec, testLogger())
	require.NoError(t, err)

	// Assert
	assert.Equal(t, http.StatusTooManyRequests, rec.Code)
	assert.Contains(t, rec.Body.String(), "rate_limit")
	assert.Contains(t, rec.Body.String(), "Too many requests")
}

// TestDeepSeekBackend_HandleStream_Passthrough verifies that SSE events are
// forwarded to the client without modification.
func TestDeepSeekBackend_HandleStream_Passthrough(t *testing.T) {
	// Arrange
	sseBody := "data: {\"type\":\"content_block_start\"}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hello\"}}\n\ndata: [DONE]\n\n"
	mockResp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(sseBody)),
	}

	backend := &DeepSeekBackend{}
	rec := httptest.NewRecorder()

	// Act
	err := backend.HandleStream(mockResp, rec, testLogger())
	require.NoError(t, err)

	// Assert — SSE headers set, body contains expected events
	assert.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", rec.Header().Get("Cache-Control"))
	assert.Equal(t, "keep-alive", rec.Header().Get("Connection"))
	assert.Equal(t, http.StatusOK, rec.Code)

	output := rec.Body.String()
	assert.Contains(t, output, "content_block_start")
	assert.Contains(t, output, "content_block_delta")
	assert.Contains(t, output, "[DONE]")
}

// TestDeepSeekBackend_HandleStream_EmptyStream verifies behavior with an
// empty upstream stream.
func TestDeepSeekBackend_HandleStream_EmptyStream(t *testing.T) {
	// Arrange
	mockResp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader("")),
	}

	backend := &DeepSeekBackend{}
	rec := httptest.NewRecorder()

	// Act
	err := backend.HandleStream(mockResp, rec, testLogger())
	require.NoError(t, err)

	// Assert
	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Empty(t, rec.Body.String())
}

// TestDeepSeekBackend_NeedsStreamTransform verifies the interface method
// returns false for the passthrough backend.
func TestDeepSeekBackend_NeedsStreamTransform(t *testing.T) {
	backend := &DeepSeekBackend{}
	assert.False(t, backend.NeedsStreamTransform())
}
