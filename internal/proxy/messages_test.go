package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRemoveRedactedThinking_EmptyMessages(t *testing.T) {
	result, count := RemoveRedactedThinking(nil)
	assert.Equal(t, 0, count)
	assert.Nil(t, result)
}

func TestRemoveRedactedThinking_NoRedactedThinking(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "Hello"},
			map[string]any{"type": "thinking", "thinking": "Hmm..."},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Len(t, result, 1)
}

func TestRemoveRedactedThinking_SingleRedactedThinking(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "redacted_thinking", "data": "secret"},
			map[string]any{"type": "text", "text": "Hello"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 1, count)
	assert.Len(t, result, 1)
	contentArr := result[0]["content"].([]any)
	assert.Len(t, contentArr, 1)
	assert.Equal(t, "text", contentArr[0].(map[string]any)["type"])
}

func TestRemoveRedactedThinking_MultipleRedactedThinking(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "redacted_thinking", "data": "a"},
			map[string]any{"type": "thinking", "thinking": "visible"},
			map[string]any{"type": "redacted_thinking", "data": "b"},
			map[string]any{"type": "text", "text": "hi"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 2, count)
	contentArr := result[0]["content"].([]any)
	assert.Len(t, contentArr, 2)
	assert.Equal(t, "thinking", contentArr[0].(map[string]any)["type"])
	assert.Equal(t, "text", contentArr[1].(map[string]any)["type"])
}

func TestRemoveRedactedThinking_MixedBlocks(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "think"},
			map[string]any{"type": "redacted_thinking", "data": "x"},
			map[string]any{"type": "tool_use", "name": "search"},
			map[string]any{"type": "text", "text": "done"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 1, count)
	contentArr := result[0]["content"].([]any)
	assert.Len(t, contentArr, 3)
	types := make([]string, len(contentArr))
	for i, b := range contentArr {
		types[i] = b.(map[string]any)["type"].(string)
	}
	assert.ElementsMatch(t, []string{"thinking", "tool_use", "text"}, types)
}

func TestRemoveRedactedThinking_StringContent(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": "hello world"},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Equal(t, "hello world", result[0]["content"])
}

func TestRemoveRedactedThinking_NilContent(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": nil},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Nil(t, result[0]["content"])
}

func TestRemoveRedactedThinking_NonAssistantRole(t *testing.T) {
	messages := []map[string]any{
		{"role": "user", "content": []any{
			map[string]any{"type": "redacted_thinking", "data": "secret"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Len(t, result[0]["content"].([]any), 1)
}

func TestRemoveRedactedThinking_MultipleMessages(t *testing.T) {
	messages := []map[string]any{
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": []any{
			map[string]any{"type": "redacted_thinking", "data": "x"},
			map[string]any{"type": "text", "text": "hello"},
		}},
		{"role": "user", "content": "bye"},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 1, count)
	assert.Len(t, result, 3)
	assistantContent := result[1]["content"].([]any)
	assert.Len(t, assistantContent, 1)
	assert.Equal(t, "text", assistantContent[0].(map[string]any)["type"])
}

func TestRemoveRedactedThinking_PreservesOriginalWhenNoChange(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "ok"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Len(t, result[0]["content"].([]any), 1)
}

func TestRemoveRedactedThinking_EmptyContentArray(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Empty(t, result[0]["content"].([]any))
}

func TestShouldKeepReasoningContent_NoToolCalls(t *testing.T) {
	messages := []map[string]any{
		{"role": "user", "content": "hello"},
		{"role": "assistant", "content": "hi there"},
	}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_NilMessages(t *testing.T) {
	assert.False(t, ShouldKeepReasoningContent(nil))
}

func TestShouldKeepReasoningContent_EmptyMessages(t *testing.T) {
	assert.False(t, ShouldKeepReasoningContent([]map[string]any{}))
}

func TestShouldKeepReasoningContent_WithMessageLevelToolCalls(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "tool_calls": []any{
			map[string]any{"id": "call_1", "type": "function"},
		}},
	}
	assert.True(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_WithContentToolUse(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "let me use a tool"},
			map[string]any{"type": "tool_use", "name": "get_weather"},
		}},
	}
	assert.True(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_WithContentToolResult(t *testing.T) {
	messages := []map[string]any{
		{"role": "user", "content": []any{
			map[string]any{"type": "tool_result", "tool_use_id": "id1", "content": "result"},
		}},
	}
	assert.True(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_MixedHistory(t *testing.T) {
	messages := []map[string]any{
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "hello"},
		{"role": "assistant", "tool_calls": []any{map[string]any{"id": "1"}}},
	}
	assert.True(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_EmptyToolCalls(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "tool_calls": []any{}},
	}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_NilToolCalls(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "tool_calls": nil},
	}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestStripReasoningContent_RemovesFields(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "reasoning_content": "Let me think...", "content": "answer"},
		{"role": "user", "content": "ok"},
	}
	count := StripReasoningContent(messages)
	assert.Equal(t, 1, count)
	_, exists := messages[0]["reasoning_content"]
	assert.False(t, exists)
	assert.Equal(t, "answer", messages[0]["content"])
}

func TestStripReasoningContent_MultipleMessages(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "reasoning_content": "think1", "content": "a1"},
		{"role": "assistant", "reasoning_content": "think2", "content": "a2"},
	}
	count := StripReasoningContent(messages)
	assert.Equal(t, 2, count)
}

func TestStripReasoningContent_NoReasoningContent(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": "answer"},
	}
	count := StripReasoningContent(messages)
	assert.Equal(t, 0, count)
}

func TestStripReasoningContent_NilMessages(t *testing.T) {
	count := StripReasoningContent(nil)
	assert.Equal(t, 0, count)
}

func TestStripReasoningContent_NilMessageInArray(t *testing.T) {
	messages := []map[string]any{
		nil,
		{"reasoning_content": "think", "content": "a"},
	}
	count := StripReasoningContent(messages)
	assert.Equal(t, 1, count)
}

func TestRemoveRedactedThinking_NilMessageInArray(t *testing.T) {
	messages := []map[string]any{nil}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Len(t, result, 1)
	assert.Nil(t, result[0])
}

func TestRemoveRedactedThinking_NonMapBlockInContent(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			"not_a_map",
			map[string]any{"type": "redacted_thinking", "data": "x"},
			map[string]any{"type": "text", "text": "hi"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 1, count)
	contentArr := result[0]["content"].([]any)
	assert.Len(t, contentArr, 2)
}

func TestShouldKeepReasoningContent_OtherToolCallsType(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "tool_calls": map[string]any{"id": "call"}},
	}
	assert.True(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_StringContent(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": "plain text response"},
	}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_NilMessageInArray(t *testing.T) {
	messages := []map[string]any{nil}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_NonMapBlockInContentArray(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			"not_a_map_string",
			map[string]any{"type": "tool_use", "name": "func"},
		}},
	}
	assert.True(t, ShouldKeepReasoningContent(messages))
}

func TestRemoveRedactedThinking_NoContentKey(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant"},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 0, count)
	assert.Len(t, result, 1)
}

func TestShouldKeepReasoningContent_NoContentKey(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant"},
	}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestShouldKeepReasoningContent_ContentIsNumber(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": float64(42)},
	}
	assert.False(t, ShouldKeepReasoningContent(messages))
}

func TestIsUnsupportedContentType(t *testing.T) {
	assert.True(t, isUnsupportedContentType("redacted_thinking"))
	assert.True(t, isUnsupportedContentType("server_tool_use"))
	assert.True(t, isUnsupportedContentType("web_search_tool_result"))
	assert.True(t, isUnsupportedContentType("code_execution_tool_result"))
	assert.True(t, isUnsupportedContentType("mcp_tool_use"))
	assert.True(t, isUnsupportedContentType("mcp_tool_result"))
	assert.True(t, isUnsupportedContentType("container_upload"))
	assert.True(t, isUnsupportedContentType("image"))
	assert.True(t, isUnsupportedContentType("document"))
	assert.True(t, isUnsupportedContentType("search_result"))
	assert.False(t, isUnsupportedContentType("text"))
	assert.False(t, isUnsupportedContentType("thinking"))
	assert.False(t, isUnsupportedContentType("tool_use"))
	assert.False(t, isUnsupportedContentType("tool_result"))
	assert.False(t, isUnsupportedContentType(42))
	assert.False(t, isUnsupportedContentType(nil))
}

func TestCountThinkingWithoutSignature_NoThinking(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "hello"},
		}},
	}
	assert.Equal(t, 0, CountThinkingWithoutSignature(messages))
}

func TestCountThinkingWithoutSignature_WithSignature(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "hmm", "signature": "sig123"},
		}},
	}
	assert.Equal(t, 0, CountThinkingWithoutSignature(messages))
}

func TestCountThinkingWithoutSignature_MissingSignature(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "hmm"},
		}},
	}
	assert.Equal(t, 1, CountThinkingWithoutSignature(messages))
}

func TestCountThinkingWithoutSignature_Mixed(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "a", "signature": "s1"},
			map[string]any{"type": "thinking", "thinking": "b"},
			map[string]any{"type": "text", "text": "hi"},
			map[string]any{"type": "thinking", "thinking": "c", "signature": "s2"},
		}},
	}
	assert.Equal(t, 1, CountThinkingWithoutSignature(messages))
}

func TestCountThinkingWithoutSignature_NilMessages(t *testing.T) {
	assert.Equal(t, 0, CountThinkingWithoutSignature(nil))
}

func TestRemoveRedactedThinking_RemovesUnsupportedTypes(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": []any{
			map[string]any{"type": "image", "source": "..."},
			map[string]any{"type": "document", "source": "..."},
			map[string]any{"type": "text", "text": "hi"},
			map[string]any{"type": "server_tool_use", "id": "x"},
		}},
	}
	result, count := RemoveRedactedThinking(messages)
	assert.Equal(t, 3, count)
	contentArr := result[0]["content"].([]any)
	assert.Len(t, contentArr, 1)
	assert.Equal(t, "text", contentArr[0].(map[string]any)["type"])
}
