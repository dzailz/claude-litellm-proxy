// Package backend defines upstream API abstractions.
//
// This file implements the DeepSeek backend — a direct passthrough to
// DeepSeek's native Anthropic-compatible API with message transformation
// logic to strip unsupported content types and manage reasoning content.
package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// DeepSeekBackend sends Anthropic-format requests to DeepSeek's native
// API endpoint. It implements the Backend interface with simple
// passthrough semantics — DeepSeek speaks the Anthropic protocol
// natively, so no format translation is needed.
type DeepSeekBackend struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
}

// Compile-time interface satisfaction check.
var _ Backend = (*DeepSeekBackend)(nil)

// PrepareRequest transforms an Anthropic-format request body for DeepSeek's
// upstream API. It applies message sanitization (redacted thinking removal,
// reasoning content stripping) and builds an HTTP request with the required
// x-api-key header and Host: api.deepseek.com.
func (b *DeepSeekBackend) PrepareRequest(ctx context.Context, body map[string]any, headers http.Header) (*http.Request, error) {
	// Apply message transformation logic to strip unsupported content.
	messagesRaw, exists := body["messages"]
	if exists {
		if msgArr, ok := messagesRaw.([]any); ok {
			messages := make([]map[string]any, len(msgArr))
			for i, m := range msgArr {
				if mm, ok := m.(map[string]any); ok {
					messages[i] = mm
				}
			}

			messages, _ = RemoveRedactedThinking(messages)
			body["messages"] = messages

			_ = StripReasoningContent(messages)
		}
	}

	modifiedBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	upstreamURL := b.BaseURL + "/v1/messages"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(modifiedBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Host = "api.deepseek.com"
	req.Header.Set("x-api-key", b.APIKey)

	// Copy incoming headers, excluding auth/host/content-length to avoid
	// leaking credentials or conflicting with our own settings.
	for key, values := range headers {
		lowerKey := strings.ToLower(key)
		if lowerKey == "x-api-key" || lowerKey == "authorization" ||
			lowerKey == "host" || lowerKey == "content-length" {
			continue
		}
		for _, v := range values {
			req.Header.Add(key, v)
		}
	}

	return req, nil
}

// NeedsStreamTransform returns false — DeepSeek speaks the Anthropic SSE
// protocol natively, so streaming events can be forwarded without
// format translation.
func (b *DeepSeekBackend) NeedsStreamTransform() bool {
	return false
}

// HandleResponse forwards a non-streaming upstream response to the client.
// Headers and body are copied directly without modification since DeepSeek
// returns Anthropic-format responses.
func (b *DeepSeekBackend) HandleResponse(resp *http.Response, w http.ResponseWriter, logger *slog.Logger) error {
	defer resp.Body.Close()

	for key, values := range resp.Header {
		for _, v := range values {
			w.Header().Add(key, v)
		}
	}
	w.WriteHeader(resp.StatusCode)

	_, err := io.Copy(w, resp.Body)
	return err
}

// HandleStream forwards an SSE streaming response from the upstream
// DeepSeek API to the client. Since DeepSeek speaks Anthropic SSE natively,
// events are copied without transformation. It sets the required SSE
// response headers and flushes after each event line.
func (b *DeepSeekBackend) HandleStream(resp *http.Response, w http.ResponseWriter, logger *slog.Logger) error {
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(resp.StatusCode)

	flusher, ok := w.(http.Flusher)
	if ok {
		flusher.Flush()
	}

	return processSSEStream(resp.Body, w)
}

// processSSEStream copies SSE events line-by-line from the upstream
// response body to the writer, flushing after each line for real-time
// delivery to the client.
func processSSEStream(body io.Reader, writer io.Writer) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	flusher, _ := writer.(http.Flusher)

	for scanner.Scan() {
		line := scanner.Text()
		if _, err := io.WriteString(writer, line+"\n"); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
	}

	return scanner.Err()
}

// ---------------------------------------------------------------------------
// Message transformation functions — extracted from internal/proxy/messages.go
// to live alongside the DeepSeek backend that uses them.
// ---------------------------------------------------------------------------

// unsupportedContentTypes lists Anthropic content block types that DeepSeek's
// API does not support and must be stripped from the request body.
var unsupportedContentTypes = map[string]bool{
	"redacted_thinking":          true,
	"server_tool_use":            true,
	"web_search_tool_result":     true,
	"code_execution_tool_result": true,
	"mcp_tool_use":               true,
	"mcp_tool_result":            true,
	"container_upload":           true,
	"image":                      true,
	"document":                   true,
	"search_result":              true,
}

// IsUnsupportedContentType reports whether the given content block type
// is not supported by the DeepSeek API and should be removed.
func IsUnsupportedContentType(t any) bool {
	s, ok := t.(string)
	if !ok {
		return false
	}
	return unsupportedContentTypes[s]
}

// RemoveRedactedThinking removes unsupported content block types (e.g.,
// redacted_thinking, server_tool_use, web_search_tool_result) from assistant
// messages. Returns the filtered message slice and the count of blocks removed.
//
// Only assistant-role messages are processed; user messages pass through
// unchanged. Messages with string content or nil content are preserved.
func RemoveRedactedThinking(messages []map[string]any) ([]map[string]any, int) {
	if len(messages) == 0 {
		return messages, 0
	}

	totalRemoved := 0
	result := make([]map[string]any, 0, len(messages))

	for _, msg := range messages {
		if msg == nil {
			result = append(result, msg)
			continue
		}

		role, _ := msg["role"].(string)
		if role != "assistant" {
			result = append(result, msg)
			continue
		}

		content, ok := msg["content"]
		if !ok {
			result = append(result, msg)
			continue
		}

		contentArr, isArr := content.([]any)
		if !isArr {
			result = append(result, msg)
			continue
		}

		if len(contentArr) == 0 {
			result = append(result, msg)
			continue
		}

		filtered := make([]any, 0, len(contentArr))
		removed := 0
		for _, block := range contentArr {
			blk, ok := block.(map[string]any)
			if !ok {
				filtered = append(filtered, block)
				continue
			}
			if IsUnsupportedContentType(blk["type"]) {
				removed++
				continue
			}
			filtered = append(filtered, block)
		}

		if removed > 0 {
			cp := copyMessage(msg)
			cp["content"] = filtered
			result = append(result, cp)
			totalRemoved += removed
		} else {
			result = append(result, msg)
		}
	}

	return result, totalRemoved
}

// ShouldKeepReasoningContent checks whether the conversation context
// contains tool calls — either at the message level (tool_calls field)
// or within content blocks (tool_use/tool_result types). When tool calls
// are present, reasoning_content should be preserved for context.
func ShouldKeepReasoningContent(messages []map[string]any) bool {
	for _, msg := range messages {
		if msg == nil {
			continue
		}

		if tc, ok := msg["tool_calls"]; ok && tc != nil {
			switch v := tc.(type) {
			case []any:
				if len(v) > 0 {
					return true
				}
			case []map[string]any:
				if len(v) > 0 {
					return true
				}
			case map[string]any:
				return true
			default:
			}
		}

		content, ok := msg["content"]
		if !ok {
			continue
		}

		contentArr, isArr := content.([]any)
		if !isArr {
			continue
		}

		for _, block := range contentArr {
			blk, ok := block.(map[string]any)
			if !ok {
				continue
			}
			t := blk["type"]
			if t == "tool_use" || t == "tool_result" {
				return true
			}
		}
	}

	return false
}

// StripReasoningContent removes the "reasoning_content" field from all
// messages (in-place mutation). Returns the count of messages that had
// the field removed. Nil messages are skipped safely.
func StripReasoningContent(messages []map[string]any) int {
	count := 0
	for _, msg := range messages {
		if msg == nil {
			continue
		}
		if _, ok := msg["reasoning_content"]; ok {
			delete(msg, "reasoning_content")
			count++
		}
	}
	return count
}

// CountThinkingWithoutSignature counts thinking content blocks in assistant
// messages that lack a cryptographic signature field. Unsigned thinking
// blocks are typically injected by proxy middleware and need to be tracked
// for debugging.
func CountThinkingWithoutSignature(messages []map[string]any) int {
	count := 0
	for _, msg := range messages {
		if msg == nil {
			continue
		}
		contentArr, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for _, block := range contentArr {
			blk, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if blk["type"] != "thinking" {
				continue
			}
			if _, ok := blk["signature"]; !ok {
				count++
			}
		}
	}
	return count
}

// copyMessage creates a shallow copy of a message map. The content block
// slices are NOT deep-copied — callers take responsibility for
// reassigning the "content" key when modifying it.
func copyMessage(m map[string]any) map[string]any {
	cp := make(map[string]any, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}
