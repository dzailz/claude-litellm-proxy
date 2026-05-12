// Package backend tests for the LiteLLM Anthropic↔OpenAI translation logic.
//
// Tests are organized into four groups:
//  1. Pure helper functions (direct unit tests)
//  2. Request translation (via PrepareRequest body inspection)
//  3. Response translation (via HandleResponse with mock responses)
//  4. Streaming translation (via HandleStream with SSE input strings)
package backend

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// Test helpers
// ============================================================================

// testBackend returns a LiteLLMBackend with dummy configuration suitable for
// unit tests that only inspect request body translation (no real HTTP calls).
func testBackend() *LiteLLMBackend {
	return &LiteLLMBackend{
		BaseURL: "http://localhost:4000",
		APIKey:  "sk-test-key",
		ModelMap: map[string]string{
			"claude-sonnet-4-20250514":  "anthropic/claude-sonnet-4-20250514",
			"claude-opus-4-20250514":    "anthropic/claude-opus-4-20250514",
			"claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet-20241022",
		},
	}
}

// parseRequestBody extracts and JSON-parses the body from a prepared *http.Request.
func parseRequestBody(t *testing.T, req *http.Request) map[string]any {
	t.Helper()
	bodyBytes, err := io.ReadAll(req.Body)
	require.NoError(t, err)
	var parsed map[string]any
	err = json.Unmarshal(bodyBytes, &parsed)
	require.NoError(t, err)
	return parsed
}

// newMockResponse creates an *http.Response with the given status code and JSON body.
func newMockResponse(statusCode int, jsonBody string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(jsonBody)),
	}
}

// parseResponseBody reads the response recorder body and unmarshals it as JSON.
func parseResponseBody(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var result map[string]any
	err := json.Unmarshal(rec.Body.Bytes(), &result)
	require.NoError(t, err)
	return result
}

// testLogger is defined in backend_test.go (shared across all backend tests).

// ============================================================================
// 1. Pure Helper Function Tests
// ============================================================================

// ---- mapModel ----

func TestMapModel_FoundInMap(t *testing.T) {
	modelMap := map[string]string{
		"claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514",
	}
	result := mapModel("claude-sonnet-4-20250514", modelMap)
	assert.Equal(t, "anthropic/claude-sonnet-4-20250514", result)
}

func TestMapModel_NotFoundInMap(t *testing.T) {
	modelMap := map[string]string{
		"claude-sonnet-4": "anthropic/claude-sonnet-4",
	}
	result := mapModel("unknown-model", modelMap)
	assert.Equal(t, "unknown-model", result)
}

func TestMapModel_EmptyMap(t *testing.T) {
	result := mapModel("any-model", map[string]string{})
	assert.Equal(t, "any-model", result)
}

func TestMapModel_EmptyKey(t *testing.T) {
	modelMap := map[string]string{"": "empty-mapped"}
	result := mapModel("", modelMap)
	assert.Equal(t, "empty-mapped", result)
}

// ---- reverseModelMap ----

func TestReverseModelMap_FoundByValue(t *testing.T) {
	modelMap := map[string]string{
		"claude-sonnet-4": "anthropic/claude-sonnet-4",
	}
	result := reverseModelMap(modelMap, "anthropic/claude-sonnet-4", "fallback")
	assert.Equal(t, "claude-sonnet-4", result)
}

func TestReverseModelMap_NotFoundReturnsOriginal(t *testing.T) {
	modelMap := map[string]string{
		"claude-sonnet-4": "anthropic/claude-sonnet-4",
	}
	result := reverseModelMap(modelMap, "unknown-model", "unknown-model")
	assert.Equal(t, "unknown-model", result)
}

func TestReverseModelMap_EmptyMap(t *testing.T) {
	result := reverseModelMap(map[string]string{}, "some-model", "some-model")
	assert.Equal(t, "some-model", result)
}

func TestReverseModelMap_MultipleMappings(t *testing.T) {
	modelMap := map[string]string{
		"claude-sonnet-4": "anthropic/claude-sonnet-4",
		"claude-opus-4":   "anthropic/claude-opus-4",
	}
	result := reverseModelMap(modelMap, "anthropic/claude-opus-4", "fallback")
	assert.Equal(t, "claude-opus-4", result)
}

// ---- mapStopReason ----

func TestMapStopReason_Stop(t *testing.T) {
	assert.Equal(t, "end_turn", mapStopReason("stop"))
}

func TestMapStopReason_ToolCalls(t *testing.T) {
	assert.Equal(t, "tool_use", mapStopReason("tool_calls"))
}

func TestMapStopReason_Length(t *testing.T) {
	assert.Equal(t, "max_tokens", mapStopReason("length"))
}

func TestMapStopReason_ContentFilter(t *testing.T) {
	assert.Equal(t, "refusal", mapStopReason("content_filter"))
}

func TestMapStopReason_UnknownPassThrough(t *testing.T) {
	assert.Equal(t, "custom_reason", mapStopReason("custom_reason"))
}

func TestMapStopReason_Empty(t *testing.T) {
	assert.Equal(t, "", mapStopReason(""))
}

// ---- formatAnthropicID ----

func TestFormatAnthropicID_ChatcmplPrefix(t *testing.T) {
	result := formatAnthropicID("chatcmpl-abc123")
	assert.Equal(t, "msg_chatcmpl-abc123", result)
}

func TestFormatAnthropicID_AlreadyMsgPrefix(t *testing.T) {
	result := formatAnthropicID("msg_abc123")
	assert.Equal(t, "msg_abc123", result)
}

func TestFormatAnthropicID_PlainID(t *testing.T) {
	result := formatAnthropicID("abc123")
	assert.Equal(t, "msg_abc123", result)
}

func TestFormatAnthropicID_Empty(t *testing.T) {
	result := formatAnthropicID("")
	assert.Equal(t, "msg_", result)
}

// ---- extractSystemContent ----

func TestExtractSystemContent_String(t *testing.T) {
	result := extractSystemContent("You are a helpful assistant")
	assert.Equal(t, "You are a helpful assistant", result)
}

func TestExtractSystemContent_ArrayOfTextBlocks(t *testing.T) {
	systemPrompt := []any{
		map[string]any{"type": "text", "text": "You are helpful."},
		map[string]any{"type": "text", "text": "Be concise."},
	}
	result := extractSystemContent(systemPrompt)
	assert.Equal(t, "You are helpful.\nBe concise.", result)
}

func TestExtractSystemContent_Nil(t *testing.T) {
	assert.Equal(t, "", extractSystemContent(nil))
}

func TestExtractSystemContent_EmptyString(t *testing.T) {
	assert.Equal(t, "", extractSystemContent(""))
}

func TestExtractSystemContent_EmptyArray(t *testing.T) {
	assert.Equal(t, "", extractSystemContent([]any{}))
}

func TestExtractSystemContent_ArrayWithNonTextBlocks(t *testing.T) {
	systemPrompt := []any{
		map[string]any{"type": "text", "text": "Hello"},
		map[string]any{"type": "image", "source": "..."},
	}
	result := extractSystemContent(systemPrompt)
	assert.Equal(t, "Hello", result)
}

func TestExtractSystemContent_ArrayWithNonMapElements(t *testing.T) {
	systemPrompt := []any{
		"not a map",
		map[string]any{"type": "text", "text": "Hello"},
	}
	result := extractSystemContent(systemPrompt)
	assert.Equal(t, "Hello", result)
}

func TestExtractSystemContent_NonStringNonArray(t *testing.T) {
	assert.Equal(t, "", extractSystemContent(42))
}

// ---- injectSystemMessage ----

func TestInjectSystemMessage_WithPrompt(t *testing.T) {
	result := injectSystemMessage(nil, "You are a bot")
	require.Len(t, result, 1)
	assert.Equal(t, "system", result[0]["role"])
	assert.Equal(t, "You are a bot", result[0]["content"])
}

func TestInjectSystemMessage_NilPrompt(t *testing.T) {
	result := injectSystemMessage(nil, nil)
	assert.Empty(t, result)
}

func TestInjectSystemMessage_EmptyStringPrompt(t *testing.T) {
	result := injectSystemMessage(nil, "")
	assert.Empty(t, result)
}

func TestInjectSystemMessage_PrependsToExisting(t *testing.T) {
	existing := []map[string]any{
		{"role": "user", "content": "Hello"},
	}
	result := injectSystemMessage(existing, "System prompt")
	require.Len(t, result, 2)
	assert.Equal(t, "system", result[0]["role"])
	assert.Equal(t, "user", result[1]["role"])
}

func TestInjectSystemMessage_ArraySystemPrompt(t *testing.T) {
	systemPrompt := []any{
		map[string]any{"type": "text", "text": "You are helpful."},
	}
	result := injectSystemMessage(nil, systemPrompt)
	require.Len(t, result, 1)
	assert.Equal(t, "system", result[0]["role"])
	assert.Equal(t, "You are helpful.", result[0]["content"])
}

// ---- translateTools ----

func TestTranslateTools_Basic(t *testing.T) {
	tools := []any{
		map[string]any{
			"name":         "get_weather",
			"description":  "Get the weather",
			"input_schema": map[string]any{"type": "object"},
		},
	}
	result := translateTools(tools)
	require.Len(t, result, 1)
	assert.Equal(t, "function", result[0]["type"])
	fn := result[0]["function"].(map[string]any)
	assert.Equal(t, "get_weather", fn["name"])
	assert.Equal(t, "Get the weather", fn["description"])
	assert.NotNil(t, fn["parameters"])
}

func TestTranslateTools_EmptyArray(t *testing.T) {
	result := translateTools([]any{})
	assert.Empty(t, result)
}

func TestTranslateTools_NilArray(t *testing.T) {
	result := translateTools(nil)
	assert.Empty(t, result)
}

func TestTranslateTools_SkipsNonMapElements(t *testing.T) {
	tools := []any{
		"not a map",
		map[string]any{"name": "valid_tool", "input_schema": map[string]any{}},
	}
	result := translateTools(tools)
	assert.Len(t, result, 1)
}

func TestTranslateTools_PreservesInputSchema(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []any{"city"},
	}
	tools := []any{
		map[string]any{"name": "search", "input_schema": schema},
	}
	result := translateTools(tools)
	require.Len(t, result, 1)
	fn := result[0]["function"].(map[string]any)
	params := fn["parameters"].(map[string]any)
	assert.Equal(t, "object", params["type"])
}

// ---- translateToolChoice ----

func TestTranslateToolChoice_Auto(t *testing.T) {
	result := translateToolChoice(map[string]any{"type": "auto"})
	assert.Equal(t, "auto", result)
}

func TestTranslateToolChoice_Any(t *testing.T) {
	result := translateToolChoice(map[string]any{"type": "any"})
	assert.Equal(t, "required", result)
}

func TestTranslateToolChoice_Tool(t *testing.T) {
	result := translateToolChoice(map[string]any{
		"type": "tool",
		"name": "get_weather",
	})
	resultMap, ok := result.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", resultMap["type"])
	fn := resultMap["function"].(map[string]any)
	assert.Equal(t, "get_weather", fn["name"])
}

func TestTranslateToolChoice_StringPassthrough(t *testing.T) {
	result := translateToolChoice("auto")
	assert.Equal(t, "auto", result)
}

func TestTranslateToolChoice_Nil(t *testing.T) {
	result := translateToolChoice(nil)
	assert.Nil(t, result)
}

func TestTranslateToolChoice_UnknownTypePassthrough(t *testing.T) {
	input := map[string]any{"type": "unknown_type"}
	result := translateToolChoice(input)
	assert.Equal(t, input, result)
}

func TestTranslateToolChoice_NonMapNonString(t *testing.T) {
	result := translateToolChoice(42)
	assert.Equal(t, 42, result)
}

// ---- normalizeToolResultContent ----

func TestNormalizeToolResultContent_String(t *testing.T) {
	result := normalizeToolResultContent("result text")
	assert.Equal(t, "result text", result)
}

func TestNormalizeToolResultContent_ArrayOfBlocks(t *testing.T) {
	content := []any{
		map[string]any{"type": "text", "text": "Part 1"},
		map[string]any{"type": "text", "text": "Part 2"},
	}
	result := normalizeToolResultContent(content)
	assert.Equal(t, "Part 1 Part 2", result)
}

func TestNormalizeToolResultContent_Nil(t *testing.T) {
	result := normalizeToolResultContent(nil)
	assert.Equal(t, "", result)
}

func TestNormalizeToolResultContent_EmptyArray(t *testing.T) {
	result := normalizeToolResultContent([]any{})
	assert.Equal(t, "", result)
}

func TestNormalizeToolResultContent_MixedArray(t *testing.T) {
	content := []any{
		map[string]any{"type": "text", "text": "A"},
		"string item",
		map[string]any{"type": "text", "text": "B"},
	}
	result := normalizeToolResultContent(content)
	assert.Equal(t, "A string item B", result)
}

func TestNormalizeToolResultContent_NonStringNonArray(t *testing.T) {
	result := normalizeToolResultContent(42)
	assert.Equal(t, 42, result)
}

// ---- translateContentBlocks (tested via PrepareRequest integration below; direct edge cases here) ----

func TestTranslateContentBlocks_EmptyArray(t *testing.T) {
	content, toolCalls, toolResultMsgs := translateContentBlocks([]any{})
	assert.Equal(t, "", content)
	assert.Nil(t, toolCalls)
	assert.Nil(t, toolResultMsgs)
}

func TestTranslateContentBlocks_NonMapBlockSkipped(t *testing.T) {
	blocks := []any{
		"not a map",
		map[string]any{"type": "text", "text": "Hello"},
	}
	content, _, _ := translateContentBlocks(blocks)
	assert.Equal(t, "Hello", content)
}

// ---- buildAnthropicContent ----

func TestBuildAnthropicContent_TextOnly(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "Hello world",
		},
	}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 1)
	assert.Equal(t, "text", result[0]["type"])
	assert.Equal(t, "Hello world", result[0]["text"])
}

func TestBuildAnthropicContent_ToolCalls(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "",
			"tool_calls": []any{
				map[string]any{
					"id": "call_1",
					"function": map[string]any{
						"name":      "get_weather",
						"arguments": `{"city":"London"}`,
					},
				},
			},
		},
	}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 1)
	assert.Equal(t, "tool_use", result[0]["type"])
	assert.Equal(t, "call_1", result[0]["id"])
	assert.Equal(t, "get_weather", result[0]["name"])
	input := result[0]["input"].(map[string]any)
	assert.Equal(t, "London", input["city"])
}

func TestBuildAnthropicContent_TextAndToolCalls(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "Let me check the weather.",
			"tool_calls": []any{
				map[string]any{
					"id": "call_1",
					"function": map[string]any{
						"name":      "get_weather",
						"arguments": "{}",
					},
				},
			},
		},
	}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 2)
	assert.Equal(t, "text", result[0]["type"])
	assert.Equal(t, "tool_use", result[1]["type"])
}

func TestBuildAnthropicContent_EmptyContentProducesEmptyBlock(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "",
		},
	}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 1)
	assert.Equal(t, "text", result[0]["type"])
	assert.Equal(t, "", result[0]["text"])
}

func TestBuildAnthropicContent_NilMessage(t *testing.T) {
	choice := map[string]any{}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 1)
	assert.Equal(t, "text", result[0]["type"])
}

func TestBuildAnthropicContent_ToolCallWithInvalidJSONArgs(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "",
			"tool_calls": []any{
				map[string]any{
					"id": "call_1",
					"function": map[string]any{
						"name":      "test",
						"arguments": "not-json",
					},
				},
			},
		},
	}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 1)
	assert.Equal(t, "tool_use", result[0]["type"])
	// Invalid JSON should fall through to empty map
	assert.NotNil(t, result[0]["input"])
}

func TestBuildAnthropicContent_ToolCallsSkipsNonMap(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "",
			"tool_calls": []any{
				"not a map",
				map[string]any{
					"id": "call_1",
					"function": map[string]any{
						"name":      "valid",
						"arguments": "{}",
					},
				},
			},
		},
	}
	result := buildAnthropicContent(choice)
	require.Len(t, result, 1)
	assert.Equal(t, "valid", result[0]["name"])
}

func TestBuildAnthropicContent_ToolCallWithoutFunction(t *testing.T) {
	choice := map[string]any{
		"message": map[string]any{
			"content": "",
			"tool_calls": []any{
				map[string]any{
					"id": "call_no_fn",
				},
			},
		},
	}
	result := buildAnthropicContent(choice)
	// No function = skipped, falls to empty text block
	require.Len(t, result, 1)
	assert.Equal(t, "text", result[0]["type"])
}

// ---- translateMessages (light direct tests; heavy tests via PrepareRequest) ----

func TestTranslateMessages_EmptyArray(t *testing.T) {
	result := translateMessages([]any{}, nil)
	assert.Empty(t, result)
}

func TestTranslateMessages_EmptyArrayWithSystem(t *testing.T) {
	result := translateMessages([]any{}, "I am a bot")
	require.Len(t, result, 1)
	assert.Equal(t, "system", result[0]["role"])
}

func TestTranslateMessages_SkipsNonMapElements(t *testing.T) {
	result := translateMessages([]any{
		"not a map",
		map[string]any{"role": "user", "content": "Hello"},
	}, nil)
	assert.Len(t, result, 1)
}

func TestTranslateMessages_UnknownContentTypePassedThrough(t *testing.T) {
	result := translateMessages([]any{
		map[string]any{"role": "user", "content": 42},
	}, nil)
	require.Len(t, result, 1)
	assert.Equal(t, 42, result[0]["content"])
}

func TestTranslateMessages_PreservesAuxiliaryFields(t *testing.T) {
	result := translateMessages([]any{
		map[string]any{
			"role":    "user",
			"name":    "John",
			"content": "Hello",
		},
	}, nil)
	require.Len(t, result, 1)
	assert.Equal(t, "John", result[0]["name"])
}

// ---- copyMessageMap ----

func TestCopyMessageMap_PreservesAllKeys(t *testing.T) {
	original := map[string]any{
		"role":    "assistant",
		"content": "hello",
		"name":    "test",
	}
	cp := copyMessageMap(original)
	assert.Equal(t, original, cp)
	assert.NotSame(t, &original, &cp)
}

func TestCopyMessageMap_EmptyMap(t *testing.T) {
	cp := copyMessageMap(map[string]any{})
	assert.Empty(t, cp)
}

// ============================================================================
// 2. Request Translation Tests (via PrepareRequest body inspection)
// ============================================================================

// ---- text content ----

func TestPrepareRequest_TranslateTextContent(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Hello, world!"},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	require.Len(t, messages, 1)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "user", msg["role"])
	assert.Equal(t, "Hello, world!", msg["content"])
}

func TestPrepareRequest_TranslateMultiBlockText(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "First paragraph."},
					map[string]any{"type": "text", "text": "Second paragraph."},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "First paragraph.\nSecond paragraph.", msg["content"])
}

// ---- system prompt ----

func TestPrepareRequest_SystemPromptString(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"system":     "You are a helpful assistant.",
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "Hello",
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	require.Len(t, messages, 2)
	assert.Equal(t, "system", messages[0].(map[string]any)["role"])
	assert.Equal(t, "You are a helpful assistant.", messages[0].(map[string]any)["content"])
	assert.Equal(t, "user", messages[1].(map[string]any)["role"])
}

func TestPrepareRequest_SystemPromptArray(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"system": []any{
			map[string]any{"type": "text", "text": "You are helpful."},
			map[string]any{"type": "text", "text": "Be concise."},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	require.Len(t, messages, 2)
	assert.Equal(t, "You are helpful.\nBe concise.", messages[0].(map[string]any)["content"])
}

func TestPrepareRequest_SystemPromptNoMessages(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":  "claude-sonnet-4-20250514",
		"system": "System only",
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	require.Len(t, messages, 1)
	assert.Equal(t, "system", messages[0].(map[string]any)["role"])
}

// ---- thinking blocks ----

func TestPrepareRequest_ThinkingBlocks(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "thinking", "thinking": "Let me analyze this."},
					map[string]any{"type": "text", "text": "Here's the answer."},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	assert.Contains(t, msg["content"], "<thinking>Let me analyze this.</thinking>")
	assert.Contains(t, msg["content"], "Here's the answer.")
}

// ---- tool_use blocks ----

func TestPrepareRequest_ToolUseBlocks(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "text", "text": "I'll look that up."},
					map[string]any{
						"type":  "tool_use",
						"id":    "toolu_123",
						"name":  "get_weather",
						"input": map[string]any{"city": "London"},
					},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "assistant", msg["role"])
	assert.Equal(t, "I'll look that up.", msg["content"])

	toolCalls := msg["tool_calls"].([]any)
	require.Len(t, toolCalls, 1)
	tc := toolCalls[0].(map[string]any)
	assert.Equal(t, "toolu_123", tc["id"])
	assert.Equal(t, "function", tc["type"])
	fn := tc["function"].(map[string]any)
	assert.Equal(t, "get_weather", fn["name"])
	assert.Contains(t, fn["arguments"], "London")
}

// ---- tool_result blocks ----

func TestPrepareRequest_ToolResultBlocks(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":         "tool_result",
						"tool_use_id":  "toolu_123",
						"content":      "Sunny, 22°C",
					},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	// tool_result blocks are split into separate "tool" role messages,
	// plus the original message is preserved (with empty content).
	// Expect 2 messages: the tool result first, then the drained user message.
	require.Len(t, messages, 2)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "tool", msg["role"])
	assert.Equal(t, "toolu_123", msg["tool_call_id"])
	assert.Equal(t, "Sunny, 22°C", msg["content"])
}

// ---- mixed blocks ----

func TestPrepareRequest_MixedContentBlocks(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "thinking", "thinking": "Hmm..."},
					map[string]any{"type": "text", "text": "Let me use a tool."},
					map[string]any{
						"type":  "tool_use",
						"id":    "toolu_1",
						"name":  "search",
						"input": map[string]any{"q": "test"},
					},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	assert.Contains(t, msg["content"], "<thinking>Hmm...</thinking>")
	assert.Contains(t, msg["content"], "Let me use a tool.")
	assert.NotNil(t, msg["tool_calls"])
}

// ---- image blocks ----

func TestPrepareRequest_ImageBlock(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "image/jpeg",
							"data":       "abc123base64",
						},
					},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	contentArr := msg["content"].([]any)
	require.Len(t, contentArr, 1)
	imgBlock := contentArr[0].(map[string]any)
	assert.Equal(t, "image_url", imgBlock["type"])
	imageURL := imgBlock["image_url"].(map[string]any)
	assert.Contains(t, imageURL["url"], "data:image/jpeg;base64,abc123base64")
}

func TestPrepareRequest_ImageWithText(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Describe this image:"},
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "image/png",
							"data":       "zzz",
						},
					},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	contentArr := msg["content"].([]any)
	require.Len(t, contentArr, 2)
	assert.Equal(t, "text", contentArr[0].(map[string]any)["type"])
	assert.Equal(t, "image_url", contentArr[1].(map[string]any)["type"])
}

func TestPrepareRequest_ImageWithoutSourceIsSkipped(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Hello"},
					map[string]any{"type": "image"},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	// Image with no source should be skipped, leaving only text
	assert.Equal(t, "Hello", msg["content"])
}

// ---- tools ----

func TestPrepareRequest_ToolTranslation(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"tools": []any{
			map[string]any{
				"name":         "get_weather",
				"description":  "Get weather",
				"input_schema": map[string]any{"type": "object"},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	tools := parsed["tools"].([]any)
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]any)
	assert.Equal(t, "function", tool["type"])
}

// ---- tool_choice ----

func TestPrepareRequest_ToolChoiceAuto(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"tool_choice": map[string]any{"type": "auto"},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, "auto", parsed["tool_choice"])
}

func TestPrepareRequest_ToolChoiceAny(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"tool_choice": map[string]any{"type": "any"},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, "required", parsed["tool_choice"])
}

// ---- model mapping ----

func TestPrepareRequest_ModelMapping(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, "anthropic/claude-sonnet-4-20250514", parsed["model"])
}

func TestPrepareRequest_ModelNotInMapPassthrough(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "unknown-model",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, "unknown-model", parsed["model"])
}

// ---- parameter translation ----

func TestPrepareRequest_ParameterTranslation(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":         "claude-sonnet-4-20250514",
		"max_tokens":    float64(500),
		"temperature":   float64(0.7),
		"top_p":         float64(0.9),
		"stop_sequences": []any{"END", "STOP"},
		"stream":        true,
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, float64(500), parsed["max_tokens"])
	assert.Equal(t, float64(0.7), parsed["temperature"])
	assert.Equal(t, float64(0.9), parsed["top_p"])
	assert.Equal(t, true, parsed["stream"])
	// stop_sequences → stop
	assert.NotNil(t, parsed["stop"])
	stopArr := parsed["stop"].([]any)
	assert.Len(t, stopArr, 2)
}

func TestPrepareRequest_NoExtraParameters(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model": "claude-sonnet-4-20250514",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	// Should not have max_tokens, temperature, etc. since not provided
	assert.NotContains(t, parsed, "max_tokens")
	assert.NotContains(t, parsed, "temperature")
	assert.NotContains(t, parsed, "stop")
}

// ---- metadata ----

func TestPrepareRequest_MetadataUserID(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"metadata": map[string]any{
			"user_id": "user-123",
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, "user-123", parsed["user"])
}

func TestPrepareRequest_MetadataWithoutUserID(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":   "claude-sonnet-4-20250514",
		"messages": []any{map[string]any{"role": "user", "content": "Hi"}},
		"metadata": map[string]any{
			"other_field": "value",
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.NotContains(t, parsed, "user")
}

// ---- string content passthrough ----

func TestPrepareRequest_StringContentPassThrough(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{"role": "user", "content": "Plain text message"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "Plain text message", msg["content"])
}

// ---- edge cases ----

func TestPrepareRequest_NilMessages(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.NotContains(t, parsed, "messages")
}

func TestPrepareRequest_EmptyMessages(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages":   []any{},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	// Empty messages without system prompt: translateMessages returns nil slice,
	// which marshals to JSON null for the "messages" key.
	assert.Nil(t, parsed["messages"])
}

func TestPrepareRequest_NonArrayMessages(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages":   "not an array",
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	assert.Equal(t, "not an array", parsed["messages"])
}

func TestPrepareRequest_NonArrayMessagesWithSystem(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"system":     "I am a bot",
		"messages":   "not an array",
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	require.Len(t, messages, 2)
	assert.Equal(t, "system", messages[0].(map[string]any)["role"])
	assert.Equal(t, "not an array", messages[1])
}

func TestPrepareRequest_NonMapMessagesWithSystemOnly(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":  "claude-sonnet-4-20250514",
		"system": "System without messages",
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	require.Len(t, messages, 1)
	assert.Equal(t, "system", messages[0].(map[string]any)["role"])
}

func TestPrepareRequest_NonMapElementInMessages(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			"string instead of map",
			map[string]any{"role": "user", "content": "Hello"},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	assert.Len(t, messages, 1) // non-map element skipped
}

func TestPrepareRequest_URLIsSet(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages":   []any{map[string]any{"role": "user", "content": "Hi"}},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)
	assert.Equal(t, "http://localhost:4000/v1/chat/completions", req.URL.String())
	assert.Equal(t, http.MethodPost, req.Method)
}

func TestPrepareRequest_AuthHeaderIsSet(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages":   []any{map[string]any{"role": "user", "content": "Hi"}},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)
	assert.Equal(t, "Bearer sk-test-key", req.Header.Get("Authorization"))
	assert.Equal(t, "application/json", req.Header.Get("Content-Type"))
}

func TestPrepareRequest_CopiesNonAuthHeaders(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages":   []any{map[string]any{"role": "user", "content": "Hi"}},
	}

	headers := http.Header{
		"X-Custom-Header": []string{"custom-value"},
		"Anthropic-Version": []string{"2023-06-01"},
		"X-Api-Key":       []string{"should-be-filtered"},
		"Authorization":   []string{"should-be-filtered"},
	}

	req, err := backend.PrepareRequest(context.Background(), body, headers)
	require.NoError(t, err)

	assert.Equal(t, "custom-value", req.Header.Get("X-Custom-Header"))
	assert.Equal(t, "2023-06-01", req.Header.Get("Anthropic-Version"))
	assert.Empty(t, req.Header.Get("X-Api-Key"))
	// Authorization should be ours, not the filtered one
	assert.Equal(t, "Bearer sk-test-key", req.Header.Get("Authorization"))
}

func TestPrepareRequest_AuxiliaryFieldsPreserved(t *testing.T) {
	backend := testBackend()
	body := map[string]any{
		"model":      "claude-sonnet-4-20250514",
		"max_tokens": float64(100),
		"messages": []any{
			map[string]any{
				"role": "user",
				"name": "my-tool",
				"content": []any{
					map[string]any{"type": "text", "text": "Call result"},
				},
			},
		},
	}

	req, err := backend.PrepareRequest(context.Background(), body, http.Header{})
	require.NoError(t, err)

	parsed := parseRequestBody(t, req)
	messages := parsed["messages"].([]any)
	msg := messages[0].(map[string]any)
	assert.Equal(t, "my-tool", msg["name"])
}

// ============================================================================
// 3. Response Translation Tests (via HandleResponse)
// ============================================================================

func TestHandleResponse_TextContent(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-abc123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "anthropic/claude-sonnet-4-20250514",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Hello! How can I help you?"
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 7,
			"total_tokens": 17
		}
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)
	assert.Equal(t, "msg_chatcmpl-abc123", result["id"])
	assert.Equal(t, "message", result["type"])
	assert.Equal(t, "assistant", result["role"])
	assert.Equal(t, "claude-sonnet-4-20250514", result["model"])
	assert.Equal(t, "end_turn", result["stop_reason"])

	content := result["content"].([]any)
	require.Len(t, content, 1)
	assert.Equal(t, "text", content[0].(map[string]any)["type"])
	assert.Equal(t, "Hello! How can I help you?", content[0].(map[string]any)["text"])
}

func TestHandleResponse_ToolCalls(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-def456",
		"object": "chat.completion",
		"model": "anthropic/claude-sonnet-4-20250514",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "",
				"tool_calls": [{
					"id": "call_abc123",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"city\":\"London\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 20,
			"completion_tokens": 15,
			"total_tokens": 35
		}
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)

	content := result["content"].([]any)
	require.Len(t, content, 1)
	block := content[0].(map[string]any)
	assert.Equal(t, "tool_use", block["type"])
	assert.Equal(t, "call_abc123", block["id"])
	assert.Equal(t, "get_weather", block["name"])

	input := block["input"].(map[string]any)
	assert.Equal(t, "London", input["city"])

	assert.Equal(t, "tool_use", result["stop_reason"])
}

func TestHandleResponse_StopReasonMappings(t *testing.T) {
	tests := []struct {
		openaiReason  string
		anthropicReason string
	}{
		{"stop", "end_turn"},
		{"tool_calls", "tool_use"},
		{"length", "max_tokens"},
		{"content_filter", "refusal"},
	}

	for _, tc := range tests {
		t.Run(tc.openaiReason, func(t *testing.T) {
			backend := testBackend()
			openaiResp := `{
				"id": "chatcmpl-test",
				"model": "anthropic/claude-sonnet-4-20250514",
				"choices": [{
					"index": 0,
					"message": {"role": "assistant", "content": "ok"},
					"finish_reason": "` + tc.openaiReason + `"
				}]
			}`

			rec := httptest.NewRecorder()
			resp := newMockResponse(http.StatusOK, openaiResp)
			err := backend.HandleResponse(resp, rec, testLogger())
			require.NoError(t, err)

			result := parseResponseBody(t, rec)
			assert.Equal(t, tc.anthropicReason, result["stop_reason"])
		})
	}
}

func TestHandleResponse_UsageMapping(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-usage",
		"model": "anthropic/claude-sonnet-4-20250514",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Test"},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 50,
			"completion_tokens": 25,
			"total_tokens": 75
		}
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)

	usage := result["usage"].(map[string]any)
	assert.Equal(t, float64(50), usage["input_tokens"])
	assert.Equal(t, float64(25), usage["output_tokens"])
	// total_tokens is not mapped
	assert.NotContains(t, usage, "total_tokens")
}

func TestHandleResponse_ModelReverseLookup(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-rev",
		"model": "anthropic/claude-sonnet-4-20250514",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hi"},
			"finish_reason": "stop"
		}]
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)
	assert.Equal(t, "claude-sonnet-4-20250514", result["model"])
}

func TestHandleResponse_ErrorPassthrough(t *testing.T) {
	backend := testBackend()
	errorResp := `{"error":{"message":"Invalid API key","type":"auth_error"}}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusUnauthorized, errorResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)
	assert.NotNil(t, result["error"])
	errObj := result["error"].(map[string]any)
	assert.Equal(t, "Invalid API key", errObj["message"])
}

func TestHandleResponse_EmptyChoicesReturnsError(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-empty",
		"model": "anthropic/claude-opus-4-20250514",
		"choices": []
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	assert.Error(t, err)
}

func TestHandleResponse_ChoicesNotAnArray(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-bad",
		"model": "test",
		"choices": "not-an-array"
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	assert.Error(t, err)
}

func TestHandleResponse_EmptyContent(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-empty-content",
		"model": "anthropic/claude-sonnet-4-20250514",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": ""},
			"finish_reason": "stop"
		}]
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)
	content := result["content"].([]any)
	require.Len(t, content, 1)
	// Empty content should produce an empty text block
	assert.Equal(t, "text", content[0].(map[string]any)["type"])
	assert.Equal(t, "", content[0].(map[string]any)["text"])
}

func TestHandleResponse_ContentTypeIsSet(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-ct",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "ok"},
			"finish_reason": "stop"
		}]
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)
	assert.Equal(t, "application/json", rec.Header().Get("Content-Type"))
}

func TestHandleResponse_StatusCodePreserved(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-sc",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "ok"},
			"finish_reason": "stop"
		}]
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusCreated, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)
	assert.Equal(t, http.StatusCreated, rec.Code)
}

func TestHandleResponse_StopSequenceIsNil(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-ss",
		"model": "test",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "ok"},
			"finish_reason": "stop"
		}]
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)
	assert.Nil(t, result["stop_sequence"])
}

func TestHandleResponse_MultipleChoicesUsesFirst(t *testing.T) {
	backend := testBackend()
	openaiResp := `{
		"id": "chatcmpl-multi",
		"model": "anthropic/claude-sonnet-4-20250514",
		"choices": [
			{
				"index": 0,
				"message": {"role": "assistant", "content": "First choice"},
				"finish_reason": "stop"
			},
			{
				"index": 1,
				"message": {"role": "assistant", "content": "Second choice"},
				"finish_reason": "stop"
			}
		]
	}`

	rec := httptest.NewRecorder()
	resp := newMockResponse(http.StatusOK, openaiResp)
	err := backend.HandleResponse(resp, rec, testLogger())
	require.NoError(t, err)

	result := parseResponseBody(t, rec)
	content := result["content"].([]any)
	assert.Equal(t, "First choice", content[0].(map[string]any)["text"])
}

// ============================================================================
// 4. Streaming Translation Tests (via HandleStream)
// ============================================================================

func TestHandleStream_TextDeltaProducesAnthropicEvents(t *testing.T) {
	backend := testBackend()
	sseInput := `data: {"id":"chatcmpl-stream1","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-stream1","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"content":" world"}}]}
data: {"id":"chatcmpl-stream1","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}
data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	// Should contain Anthropic SSE events
	assert.Contains(t, output, "event: message_start")
	assert.Contains(t, output, "event: content_block_start")
	assert.Contains(t, output, "event: content_block_delta")
	assert.Contains(t, output, "text_delta")
	assert.Contains(t, output, "Hello")
	assert.Contains(t, output, " world")
	assert.Contains(t, output, "event: content_block_stop")
	assert.Contains(t, output, "event: message_delta")
	assert.Contains(t, output, "event: message_stop")
}

func TestHandleStream_ToolCallDeltaProducesToolUseEvents(t *testing.T) {
	backend := testBackend()
	sseInput := `data: {"id":"chatcmpl-tool","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":""}}]}}]}
data: {"id":"chatcmpl-tool","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}
data: {"id":"chatcmpl-tool","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"London\"}"}}]}}]}
data: {"id":"chatcmpl-tool","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":5}}
data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	// Should contain tool_use events
	assert.Contains(t, output, "event: message_start")
	assert.Contains(t, output, "event: content_block_start")
	assert.Contains(t, output, `"type":"tool_use"`)
	assert.Contains(t, output, "input_json_delta")
	assert.Contains(t, output, "event: message_stop")
}

func TestHandleStream_DoneMarkerEndsStream(t *testing.T) {
	backend := testBackend()
	sseInput := `data: {"id":"chatcmpl-done","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{"content":"Hi"}}]}
data: {"id":"chatcmpl-done","model":"anthropic/claude-sonnet-4-20250514","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}
data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	assert.Contains(t, output, "event: message_stop")
	// [DONE] should not appear in output (Anthropic SSE format)
	assert.NotContains(t, output, "[DONE]")
}

func TestHandleStream_EmptyStreamStillEmitsEvents(t *testing.T) {
	backend := testBackend()
	sseInput := `data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	// Should still emit message_start and message_stop for the empty stream
	assert.Contains(t, output, "event: message_stop")
}

func TestHandleStream_MalformedChunksHandledGracefully(t *testing.T) {
	backend := testBackend()
	// Non-JSON data line and a valid one
	sseInput := `data: this-is-not-json
data: {"id":"valid","model":"test","choices":[{"index":0,"delta":{"content":"Hello"}}]}
data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	// Should still process the valid chunk after the malformed one
	assert.Contains(t, output, "Hello")
	assert.Contains(t, output, "event: message_stop")
}

func TestHandleStream_NonDataLinesIgnored(t *testing.T) {
	backend := testBackend()
	sseInput := `event: some-event
data: {"id":"test","model":"test","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	assert.Contains(t, output, "Hello")
}

func TestHandleStream_NoChoicesChunkIsSkipped(t *testing.T) {
	backend := testBackend()
	sseInput := `data: {"id":"test","model":"test","choices":[]}
data: {"id":"test","model":"test","choices":[{"index":0,"delta":{"content":"Hello"}}]}
data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	output := rec.Body.String()
	assert.Contains(t, output, "Hello")
}

func TestHandleStream_SSEHeadersAreSet(t *testing.T) {
	backend := testBackend()
	sseInput := `data: [DONE]
`

	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(sseInput)),
	}

	err := backend.HandleStream(resp, rec, testLogger())
	require.NoError(t, err)

	assert.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", rec.Header().Get("Cache-Control"))
	assert.Equal(t, "keep-alive", rec.Header().Get("Connection"))
}

// ============================================================================
// 5. Interface Compliance
// ============================================================================

func TestLiteLLMBackend_ImplementsBackend(t *testing.T) {
	// Verify that the compile-time check is correct by creating an instance
	var b Backend = &LiteLLMBackend{
		BaseURL: "http://localhost:4000",
		APIKey:  "sk-test",
	}
	assert.NotNil(t, b)
}

func TestLiteLLMBackend_NeedsStreamTransform(t *testing.T) {
	backend := testBackend()
	assert.True(t, backend.NeedsStreamTransform())
}
