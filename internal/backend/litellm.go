// Package backend defines the upstream API abstraction layer.
package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// LiteLLMBackend translates Anthropic-format requests to OpenAI format
// for proxying through a LiteLLM server. It implements the Backend interface.
//
// LiteLLM accepts the OpenAI /v1/chat/completions format, so this backend
// performs a full Anthropic → OpenAI request translation and (when
// fully implemented) translates responses back to Anthropic format.
//
// Compile-time interface satisfaction check.
var _ Backend = (*LiteLLMBackend)(nil)

// LiteLLMBackend holds the configuration and HTTP client for communicating
// with a LiteLLM proxy server.
type LiteLLMBackend struct {
	// BaseURL is the LiteLLM proxy endpoint, e.g., "http://localhost:4000".
	// The /v1/chat/completions path is appended by PrepareRequest.
	BaseURL string

	// APIKey is the LiteLLM virtual key used for authentication.
	APIKey string

	// ModelMap translates Claude model names to LiteLLM model identifiers
	// (e.g., "claude-sonnet-4-20250514" → "anthropic/claude-sonnet-4-20250514").
	// If a model is not found in the map, the original name is used as-is.
	ModelMap map[string]string

	// Client is the HTTP client used to execute prepared requests.
	Client *http.Client
}

// PrepareRequest transforms an Anthropic-format request body into an
// OpenAI-format HTTP request targeting the LiteLLM proxy.
//
// Translation steps:
//  1. Extract top-level "system" field → prepend system message to messages array
//  2. Translate Anthropic messages (content blocks, tool_use, tool_result) → OpenAI messages
//  3. Translate tools (input_schema → parameters), tool_choice, model, and parameters
//  4. Construct POST request to {BaseURL}/v1/chat/completions
//
// The returned *http.Request is ready for execution via b.Client.Do.
func (b *LiteLLMBackend) PrepareRequest(ctx context.Context, body map[string]any, headers http.Header) (*http.Request, error) {
	openaiBody := make(map[string]any)

	// Extract system prompt (Anthropic top-level field)
	systemPrompt := body["system"]

	// 1. Translate messages
	if messagesRaw, ok := body["messages"]; ok {
		if msgArr, ok := messagesRaw.([]any); ok {
			openaiBody["messages"] = translateMessages(msgArr, systemPrompt)
		} else {
			// Non-array messages: pass through as-is, but still inject system if present
			if systemPrompt != nil {
				msgs := injectSystemMessage(nil, systemPrompt)
				// Merge with non-array value: wrap in array with system first
				allMsgs := make([]any, 0, len(msgs)+1)
				for _, m := range msgs {
					allMsgs = append(allMsgs, m)
				}
				allMsgs = append(allMsgs, messagesRaw)
				openaiBody["messages"] = allMsgs
			} else {
				openaiBody["messages"] = messagesRaw
			}
		}
	} else if systemPrompt != nil {
		// No messages but system prompt exists — inject system message
		msgs := injectSystemMessage(nil, systemPrompt)
		allMsgs := make([]any, len(msgs))
		for i, m := range msgs {
			allMsgs[i] = m
		}
		openaiBody["messages"] = allMsgs
	}

	// 2. Translate tools
	if toolsRaw, ok := body["tools"]; ok {
		if toolsArr, ok := toolsRaw.([]any); ok {
			openaiBody["tools"] = translateTools(toolsArr)
		}
	}

	// 3. Translate tool_choice
	if toolChoiceRaw, ok := body["tool_choice"]; ok {
		openaiBody["tool_choice"] = translateToolChoice(toolChoiceRaw)
	}

	// 4. Translate model
	if modelRaw, ok := body["model"]; ok {
		if model, ok := modelRaw.(string); ok {
			openaiBody["model"] = mapModel(model, b.ModelMap)
		} else {
			openaiBody["model"] = modelRaw
		}
	}

	// 5. Translate parameters
	if v, ok := body["max_tokens"]; ok {
		openaiBody["max_tokens"] = v
	}
	if v, ok := body["temperature"]; ok {
		openaiBody["temperature"] = v
	}
	if v, ok := body["top_p"]; ok {
		openaiBody["top_p"] = v
	}
	// top_k is intentionally dropped — OpenAI does not support it
	if v, ok := body["stop_sequences"]; ok {
		openaiBody["stop"] = v
	}
	if v, ok := body["stream"]; ok {
		openaiBody["stream"] = v
	}

	// metadata.user_id → user
	if metadataRaw, ok := body["metadata"]; ok {
		if metadata, ok := metadataRaw.(map[string]any); ok {
			if userID, ok := metadata["user_id"]; ok {
				openaiBody["user"] = userID
			}
		}
	}

	// Marshal to JSON
	bodyBytes, err := json.Marshal(openaiBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal OpenAI request body: %w", err)
	}

	// Construct HTTP request
	url := b.BaseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create upstream request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+b.APIKey)

	// Copy non-auth client headers (same pattern as handler.go lines 128-137)
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

// NeedsStreamTransform reports that LiteLLM responses require SSE format
// translation (OpenAI SSE → Anthropic SSE).
func (b *LiteLLMBackend) NeedsStreamTransform() bool {
	return true
}

// HandleResponse translates a non-streaming OpenAI Chat Completions response
// back to Anthropic Messages format and writes it to the client.
//
// Translation covers all required fields: id, type, role, model, content
// (text and tool_use blocks), stop_reason, and usage. Upstream headers
// are intentionally not forwarded — Anthropic clients expect only the
// Anthropic-format JSON body with standard headers.
func (b *LiteLLMBackend) HandleResponse(resp *http.Response, w http.ResponseWriter, logger *slog.Logger) error {
	defer resp.Body.Close()

	// Read the full upstream response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.Error("failed to read upstream response body", "error", err)
		return fmt.Errorf("failed to read upstream response body: %w", err)
	}

	// Parse the JSON response
	var rawResp map[string]any
	if err := json.Unmarshal(bodyBytes, &rawResp); err != nil {
		logger.Error("failed to parse upstream response JSON", "error", err)
		return fmt.Errorf("failed to parse upstream response JSON: %w", err)
	}

	// If upstream returned an error (e.g., {"error":{"message":"..."}}),
	// pass it through to the client as-is
	if _, hasError := rawResp["error"]; hasError {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		return json.NewEncoder(w).Encode(rawResp)
	}

	// Validate choices array — Anthropic response requires at least one choice
	choicesRaw, ok := rawResp["choices"].([]any)
	if !ok || len(choicesRaw) == 0 {
		logger.Error("upstream response has no choices", "status_code", resp.StatusCode)
		return fmt.Errorf("upstream response has no choices")
	}

	firstChoice, ok := choicesRaw[0].(map[string]any)
	if !ok {
		return fmt.Errorf("invalid choice format in upstream response")
	}

	// Build the Anthropic-format response
	anthropicResp := make(map[string]any)

	// id — reformat with "msg_" prefix (OpenAI uses "chatcmpl-...")
	if id, ok := rawResp["id"].(string); ok {
		anthropicResp["id"] = formatAnthropicID(id)
	}

	// type — always "message" for non-streaming responses
	anthropicResp["type"] = "message"

	// role — always "assistant" for the model's response
	anthropicResp["role"] = "assistant"

	// model — reverse mapping back to the original Anthropic model name
	openaiModel, _ := rawResp["model"].(string)
	anthropicResp["model"] = reverseModelMap(b.ModelMap, openaiModel, openaiModel)

	// content — text and tool_use blocks built from the OpenAI choice
	anthropicResp["content"] = buildAnthropicContent(firstChoice)

	// stop_reason — translate OpenAI finish_reason to Anthropic stop_reason
	finishReason, _ := firstChoice["finish_reason"].(string)
	anthropicResp["stop_reason"] = mapStopReason(finishReason)
	anthropicResp["stop_sequence"] = nil

	// usage — remap prompt_tokens→input_tokens, completion_tokens→output_tokens
	if usageRaw, ok := rawResp["usage"].(map[string]any); ok {
		usage := map[string]any{}
		if v, ok := usageRaw["prompt_tokens"]; ok {
			usage["input_tokens"] = v
		}
		if v, ok := usageRaw["completion_tokens"]; ok {
			usage["output_tokens"] = v
		}
		if len(usage) > 0 {
			anthropicResp["usage"] = usage
		}
	}

	// Write the Anthropic response — only set Content-Type, no upstream headers
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	return json.NewEncoder(w).Encode(anthropicResp)
}

// HandleStream translates an OpenAI SSE streaming response to Anthropic
// SSE format in real time. It reads the upstream response line-by-line
// using a bufio.Scanner, parses each "data:" line as an OpenAI chat
// completion chunk, and emits the equivalent Anthropic SSE events to the
// response writer.
//
// Translation handles the full streaming lifecycle:
//   - message_start (emitted on first content/tool_use delta)
//   - content_block_start / content_block_delta / content_block_stop
//     for text and tool_use blocks
//   - message_delta with stop_reason mapping and output_tokens
//   - message_stop
func (b *LiteLLMBackend) HandleStream(resp *http.Response, w http.ResponseWriter, logger *slog.Logger) error {
	defer resp.Body.Close()

	// Set SSE response headers matching Anthropic SSE expectations
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(resp.StatusCode)

	flusher, ok := w.(http.Flusher)
	if ok {
		flusher.Flush()
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	state := &streamState{}

	for scanner.Scan() {
		line := scanner.Text()

		// Only process "data:" prefixed lines (skip SSE comments,
		// event: lines, and empty lines)
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		payload := strings.TrimSpace(line[5:]) // strip "data:" prefix

		// [DONE] marker signals end of stream from OpenAI
		if payload == "[DONE]" {
			if !state.finishEmitted {
				if err := emitFinalEvents(w, flusher, state); err != nil {
					return fmt.Errorf("failed to emit final SSE events: %w", err)
				}
			}
			continue
		}

		// Parse the JSON payload
		var chunk map[string]any
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			logger.Debug("skipping malformed SSE chunk", "error", err)
			continue
		}

		if err := translateStreamChunk(chunk, state, w, flusher, b.ModelMap); err != nil {
			return fmt.Errorf("failed to translate streaming chunk: %w", err)
		}
	}

	return scanner.Err()
}

// ---------------------------------------------------------------------------
// Stream translation helpers
// ---------------------------------------------------------------------------

// streamState tracks the progress of translating an OpenAI SSE stream to
// Anthropic SSE events. It maintains the current content block type and
// prevents double emission of terminal events.
type streamState struct {
	messageID         string
	model             string
	hasStartedMessage bool
	currentBlockType  string // "", "text", "tool_use"
	currentBlockIdx   int
	finishEmitted     bool
	finishReason      string
	outputTokens      int
}

// writeSSEEvent writes a single Anthropic SSE event to the writer and
// flushes immediately for real-time delivery to the client.
func writeSSEEvent(w io.Writer, flusher http.Flusher, eventType string, data any) error {
	if _, err := io.WriteString(w, "event: "+eventType+"\n"); err != nil {
		return err
	}
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: "+string(dataBytes)+"\n\n"); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

// writeMessageStart emits the Anthropic message_start event. The content
// array is empty (populated incrementally by subsequent deltas) and usage
// token counts are zero (filled in by the terminal message_delta).
func writeMessageStart(w io.Writer, flusher http.Flusher, state *streamState) error {
	state.hasStartedMessage = true
	return writeSSEEvent(w, flusher, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":      state.messageID,
			"type":    "message",
			"role":    "assistant",
			"model":   state.model,
			"content": []any{},
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})
}

// emitFinalEvents closes any open content block and emits the terminal
// Anthropic SSE events: message_delta (with stop_reason and usage) then
// message_stop. If no message was started (empty stream), a minimal
// message_start is emitted first so the event sequence is valid.
func emitFinalEvents(w io.Writer, flusher http.Flusher, state *streamState) error {
	if !state.hasStartedMessage {
		if err := writeMessageStart(w, flusher, state); err != nil {
			return err
		}
	}

	// Close the current content block if one is still open
	if state.currentBlockType != "" {
		if err := writeSSEEvent(w, flusher, "content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": state.currentBlockIdx,
		}); err != nil {
			return err
		}
	}

	// Emit message_delta with mapped stop_reason and output token count
	stopReason := mapStopReason(state.finishReason)
	if err := writeSSEEvent(w, flusher, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": state.outputTokens,
		},
	}); err != nil {
		return err
	}

	// Emit message_stop
	if err := writeSSEEvent(w, flusher, "message_stop", map[string]any{
		"type": "message_stop",
	}); err != nil {
		return err
	}

	state.finishEmitted = true
	return nil
}

// translateStreamChunk processes a single OpenAI chat completion chunk and
// emits the equivalent Anthropic SSE events in real time. It handles:
//   - Content text deltas (delta.content → text_delta)
//   - Tool call deltas (delta.tool_calls → tool_use block + input_json_delta)
//   - Finish reason (choice.finish_reason → terminal message_delta/message_stop)
func translateStreamChunk(chunk map[string]any, state *streamState, w io.Writer, flusher http.Flusher, modelMap map[string]string) error {
	// Capture model and message ID from the first available chunk (every
	// OpenAI SSE chunk carries these fields)
	if state.model == "" {
		if m, ok := chunk["model"].(string); ok {
			state.model = reverseModelMap(modelMap, m, m)
		}
	}
	if state.messageID == "" {
		if id, ok := chunk["id"].(string); ok {
			state.messageID = formatAnthropicID(id)
		}
	}

	choices, ok := chunk["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}

	firstChoice, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}

	delta, _ := firstChoice["delta"].(map[string]any)

	// ---- Process content deltas (text and tool calls) ----

	if delta != nil {
		// Text content delta — OpenAI sends delta.content as incremental
		// string tokens. Translate to Anthropic text_delta events.
		if content, ok := delta["content"].(string); ok && content != "" {
			if !state.hasStartedMessage {
				if err := writeMessageStart(w, flusher, state); err != nil {
					return err
				}
			}
			// If switching from tool_use to text, close the tool_use block first
			if state.currentBlockType != "text" {
				if state.currentBlockType != "" {
					if err := writeSSEEvent(w, flusher, "content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": state.currentBlockIdx,
					}); err != nil {
						return err
					}
					state.currentBlockIdx++
				}
				if err := writeSSEEvent(w, flusher, "content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": state.currentBlockIdx,
					"content_block": map[string]any{
						"type": "text",
						"text": "",
					},
				}); err != nil {
					return err
				}
				state.currentBlockType = "text"
			}
			if err := writeSSEEvent(w, flusher, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": state.currentBlockIdx,
				"delta": map[string]any{
					"type": "text_delta",
					"text": content,
				},
			}); err != nil {
				return err
			}
		}

		// Tool call deltas — OpenAI streams tool calls incrementally:
		// the first chunk for a tool has id and function.name, and
		// subsequent chunks accumulate function.arguments as partial JSON.
		if toolCalls, ok := delta["tool_calls"].([]any); ok {
			for _, tc := range toolCalls {
				tcMap, ok := tc.(map[string]any)
				if !ok {
					continue
				}

				id, hasID := tcMap["id"].(string)
				fn, hasFn := tcMap["function"].(map[string]any)

				// First appearance of a tool call: has an id field.
				// Start a new tool_use content block.
				if hasID {
					if !state.hasStartedMessage {
						if err := writeMessageStart(w, flusher, state); err != nil {
							return err
						}
					}
					// Close the previous content block
					if state.currentBlockType != "" {
						if err := writeSSEEvent(w, flusher, "content_block_stop", map[string]any{
							"type":  "content_block_stop",
							"index": state.currentBlockIdx,
						}); err != nil {
							return err
						}
						state.currentBlockIdx++
					}
					name := ""
					if hasFn {
						name, _ = fn["name"].(string)
					}
					state.currentBlockType = "tool_use"

					if err := writeSSEEvent(w, flusher, "content_block_start", map[string]any{
						"type":  "content_block_start",
						"index": state.currentBlockIdx,
						"content_block": map[string]any{
							"type":  "tool_use",
							"id":    id,
							"name":  name,
							"input": map[string]any{},
						},
					}); err != nil {
						return err
					}
				}

				// Emit accumulated arguments as input_json_delta
				var args string
				if hasFn {
					args, _ = fn["arguments"].(string)
				}
				if args == "" {
					// Some providers put arguments at the tool_call level
					if a, ok := tcMap["arguments"].(string); ok {
						args = a
					}
				}
				if args != "" {
					if !state.hasStartedMessage {
						if err := writeMessageStart(w, flusher, state); err != nil {
							return err
						}
					}
					// Edge case: arguments without a prior id chunk
					if state.currentBlockType == "" {
						state.currentBlockType = "tool_use"
						if err := writeSSEEvent(w, flusher, "content_block_start", map[string]any{
							"type":  "content_block_start",
							"index": state.currentBlockIdx,
							"content_block": map[string]any{
								"type":  "tool_use",
								"id":    "",
								"name":  "",
								"input": map[string]any{},
							},
						}); err != nil {
							return err
						}
					}
					if err := writeSSEEvent(w, flusher, "content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": state.currentBlockIdx,
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": args,
						},
					}); err != nil {
						return err
					}
				}
			}
		}
	}

	// ---- Process finish reason (terminal chunk) ----
	//
	// OpenAI sends finish_reason on the choice object (not inside delta).
	// This may coexist with content/tool_call deltas (the last token and
	// finish_reason arrive in the same chunk), so we process deltas first
	// and terminal events last.
	if fr, ok := firstChoice["finish_reason"].(string); ok && fr != "" && !state.finishEmitted {
		state.finishReason = fr

		// Capture completion token count from the final chunk's usage
		if usageRaw, ok := chunk["usage"].(map[string]any); ok {
			if ct, ok := usageRaw["completion_tokens"]; ok {
				switch v := ct.(type) {
				case float64:
					state.outputTokens = int(v)
				case int:
					state.outputTokens = v
				}
			}
		}

		if err := emitFinalEvents(w, flusher, state); err != nil {
			return err
		}
	}

	return nil
}

// --- Helper Functions ---

// translateMessages converts Anthropic-format messages to OpenAI format.
// It prepends a system message derived from the systemPrompt, then iterates
// over each message, translating content blocks and splitting tool_result
// blocks into their own independent "tool" role messages.
func translateMessages(anthropicMessages []any, systemPrompt any) []map[string]any {
	var result []map[string]any

	// Prepend system message if a system prompt is provided
	result = injectSystemMessage(result, systemPrompt)

	for _, msg := range anthropicMessages {
		mm, ok := msg.(map[string]any)
		if !ok {
			// Non-map elements are skipped (cannot translate)
			continue
		}

		role, _ := mm["role"].(string)
		content := mm["content"]

		// String content: pass through as-is
		if str, ok := content.(string); ok {
			newMsg := copyMessageMap(mm)
			// Keep other fields (like name) but use string content
			newMsg["content"] = str
			result = append(result, newMsg)
			continue
		}

		// Array content: translate block by block
		if blocks, ok := content.([]any); ok {
			msgContent, toolCalls, toolResultMsgs := translateContentBlocks(blocks)

			// Tool result messages are split into separate "tool" role messages
			result = append(result, toolResultMsgs...)

			newMsg := map[string]any{
				"role": role,
			}

			// Copy auxiliary fields from the original message (e.g., "name" for user messages)
			for _, key := range []string{"name"} {
				if val, ok := mm[key]; ok {
					newMsg[key] = val
				}
			}

			if len(toolCalls) > 0 {
				newMsg["tool_calls"] = toolCalls
			}

			// Determine content format: string for text-only, array for multimodal
			switch c := msgContent.(type) {
			case string:
				if c != "" {
					newMsg["content"] = c
				} else if len(toolCalls) == 0 {
					newMsg["content"] = ""
				}
			case []any:
				newMsg["content"] = c
			}

			result = append(result, newMsg)
			continue
		}

		// Unknown content type: copy message as-is
		newMsg := copyMessageMap(mm)
		result = append(result, newMsg)
	}

	return result
}

// translateContentBlocks processes an array of Anthropic content blocks and
// returns translated OpenAI components:
//   - content: either a string (text+thinking blocks joined) or []any (multimodal with images)
//   - toolCalls: tool_use blocks converted to OpenAI tool_calls format
//   - toolResultMsgs: tool_result blocks split into separate "tool" role messages
func translateContentBlocks(blocks []any) (content any, toolCalls []any, toolResultMsgs []map[string]any) {
	var textParts []string
	var imageBlocks []any

	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			continue
		}

		blockType, _ := blk["type"].(string)

		switch blockType {
		case "text":
			if text, ok := blk["text"].(string); ok {
				textParts = append(textParts, text)
			}

		case "thinking":
			// Wrap thinking content in XML-style tags
			if thinking, ok := blk["thinking"].(string); ok {
				textParts = append(textParts, "<thinking>"+thinking+"</thinking>")
			}

		case "tool_use":
			id, _ := blk["id"].(string)
			name, _ := blk["name"].(string)
			input := blk["input"]

			// JSON-stringify the input for the arguments field
			argsStr := "{}"
			if input != nil {
				if argsBytes, err := json.Marshal(input); err == nil {
					argsStr = string(argsBytes)
				}
			}

			toolCalls = append(toolCalls, map[string]any{
				"id":   id,
				"type": "function",
				"function": map[string]any{
					"name":      name,
					"arguments": argsStr,
				},
			})

		case "tool_result":
			toolUseID, _ := blk["tool_use_id"].(string)
			resultContent := normalizeToolResultContent(blk["content"])

			toolResultMsgs = append(toolResultMsgs, map[string]any{
				"role":         "tool",
				"tool_call_id": toolUseID,
				"content":      resultContent,
			})

		case "image":
			// Convert Anthropic image block to OpenAI image_url block
			source, ok := blk["source"].(map[string]any)
			if !ok {
				continue
			}
			mediaType, _ := source["media_type"].(string)
			data, _ := source["data"].(string)
			if mediaType != "" && data != "" {
				url := "data:" + mediaType + ";base64," + data
				imageBlocks = append(imageBlocks, map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": url,
					},
				})
			}
		}
	}

	textContent := strings.Join(textParts, "\n")

	// Determine content format: string for text-only, array for multimodal
	if len(imageBlocks) > 0 {
		var contentBlocks []any
		if textContent != "" {
			contentBlocks = append(contentBlocks, map[string]any{
				"type": "text",
				"text": textContent,
			})
		}
		contentBlocks = append(contentBlocks, imageBlocks...)
		content = contentBlocks
	} else {
		content = textContent
	}

	return content, toolCalls, toolResultMsgs
}

// normalizeToolResultContent flattens tool_result content to a string.
// Anthropic tool_result content can be a string or an array of text blocks.
func normalizeToolResultContent(raw any) any {
	if raw == nil {
		return ""
	}
	if str, ok := raw.(string); ok {
		return str
	}
	if arr, ok := raw.([]any); ok {
		var parts []string
		for _, item := range arr {
			if itemBlk, ok := item.(map[string]any); ok {
				if text, ok := itemBlk["text"].(string); ok {
					parts = append(parts, text)
				}
			} else if str, ok := item.(string); ok {
				parts = append(parts, str)
			}
		}
		return strings.Join(parts, " ")
	}
	return raw
}

// translateTools converts Anthropic tools format to OpenAI tools format.
// Anthropic: [{"name":"...","description":"...","input_schema":{...}}]
// OpenAI:    [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]
func translateTools(anthropicTools []any) []map[string]any {
	result := make([]map[string]any, 0, len(anthropicTools))

	for _, tool := range anthropicTools {
		t, ok := tool.(map[string]any)
		if !ok {
			continue
		}

		name, _ := t["name"].(string)
		description, _ := t["description"].(string)
		inputSchema := t["input_schema"]

		result = append(result, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        name,
				"description": description,
				"parameters":  inputSchema,
			},
		})
	}

	return result
}

// translateToolChoice converts Anthropic tool_choice format to OpenAI.
//
// Mappings:
//
//	Anthropic {"type":"auto"}       → OpenAI "auto"
//	Anthropic {"type":"any"}        → OpenAI "required"
//	Anthropic {"type":"tool","name":"x"} → OpenAI {"type":"function","function":{"name":"x"}}
//	Any other value passed through as-is
func translateToolChoice(choice any) any {
	if choice == nil {
		return nil
	}

	// String value: pass through
	if _, ok := choice.(string); ok {
		return choice
	}

	cm, ok := choice.(map[string]any)
	if !ok {
		return choice
	}

	choiceType, _ := cm["type"].(string)
	switch choiceType {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "tool":
		name, _ := cm["name"].(string)
		return map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": name,
			},
		}
	default:
		return choice
	}
}

// mapModel looks up the given model name in the ModelMap and returns the
// mapped value. If the model is not found, the original name is returned.
func mapModel(model string, modelMap map[string]string) string {
	if mapped, ok := modelMap[model]; ok {
		return mapped
	}
	return model
}

// extractSystemContent converts the Anthropic top-level "system" field
// into a plain string. It handles both string and array-of-text-blocks formats.
func extractSystemContent(systemPrompt any) string {
	if systemPrompt == nil {
		return ""
	}

	// String format: "system": "You are a helpful assistant"
	if str, ok := systemPrompt.(string); ok {
		return str
	}

	// Array format: "system": [{"type":"text","text":"You are helpful"}, ...]
	if arr, ok := systemPrompt.([]any); ok {
		var parts []string
		for _, item := range arr {
			blk, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if blkType, _ := blk["type"].(string); blkType == "text" {
				if text, ok := blk["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	}

	return ""
}

// injectSystemMessage prepends a system message to the result slice if a
// system prompt is provided. The system message is placed at the beginning
// of the slice so it appears first in the OpenAI messages array.
func injectSystemMessage(result []map[string]any, systemPrompt any) []map[string]any {
	sysContent := extractSystemContent(systemPrompt)
	if sysContent == "" {
		return result
	}
	return append([]map[string]any{{
		"role":    "system",
		"content": sysContent,
	}}, result...)
}

// copyMessageMap creates a shallow copy of a message map, preserving all
// non-content fields like "name", "role", etc.
func copyMessageMap(m map[string]any) map[string]any {
	cp := make(map[string]any, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

// --- Response Translation Helpers ---

// formatAnthropicID ensures the response ID uses the "msg_" prefix expected
// by Anthropic clients. OpenAI responses use "chatcmpl-..." format.
func formatAnthropicID(openaiID string) string {
	if strings.HasPrefix(openaiID, "msg_") {
		return openaiID
	}
	return "msg_" + openaiID
}

// reverseModelMap finds the original Anthropic model name that maps to the
// given OpenAI model name. If the OpenAI model appears as a value in the
// ModelMap, the corresponding key is returned. Otherwise, the originalModel
// fallback is used (which should be the OpenAI model itself when no mapping
// was applied during the request translation).
func reverseModelMap(modelMap map[string]string, openaiModel string, originalModel string) string {
	for key, val := range modelMap {
		if val == openaiModel {
			return key
		}
	}
	return originalModel
}

// mapStopReason translates OpenAI finish_reason values to their Anthropic
// stop_reason equivalents.
//
// Mappings:
//
//	stop           → end_turn
//	tool_calls     → tool_use
//	length         → max_tokens
//	content_filter → refusal
func mapStopReason(finishReason string) string {
	switch finishReason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "content_filter":
		return "refusal"
	default:
		return finishReason
	}
}

// buildAnthropicContent constructs the Anthropic content array from an OpenAI
// choice. Text content is placed first, followed by tool_use blocks derived
// from tool_calls. Tool call arguments are JSON-parsed back from the
// string representation into native objects.
//
// If the resulting content array would be empty (no text, no tool calls),
// a single empty text block is included per the Anthropic API spec.
func buildAnthropicContent(choice map[string]any) []map[string]any {
	var content []map[string]any

	message, _ := choice["message"].(map[string]any)
	if message == nil {
		return []map[string]any{{"type": "text", "text": ""}}
	}

	// Process text content — add as the first block if non-empty
	if textContent, ok := message["content"].(string); ok && textContent != "" {
		content = append(content, map[string]any{
			"type": "text",
			"text": textContent,
		})
	}

	// Process tool_calls — each becomes a tool_use block
	if toolCalls, ok := message["tool_calls"].([]any); ok {
		for _, tc := range toolCalls {
			tcMap, ok := tc.(map[string]any)
			if !ok {
				continue
			}

			id, _ := tcMap["id"].(string)

			fn, _ := tcMap["function"].(map[string]any)
			if fn == nil {
				continue
			}

			name, _ := fn["name"].(string)
			argsStr, _ := fn["arguments"].(string)

			// Parse the arguments JSON string back into an object
			var input any
			if argsStr != "" {
				var parsed any
				if err := json.Unmarshal([]byte(argsStr), &parsed); err == nil {
					input = parsed
				} else {
					input = map[string]any{}
				}
			} else {
				input = map[string]any{}
			}

			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    id,
				"name":  name,
				"input": input,
			})
		}
	}

	// If no blocks were produced, include an empty text block
	if len(content) == 0 {
		content = append(content, map[string]any{
			"type": "text",
			"text": "",
		})
	}

	return content
}
