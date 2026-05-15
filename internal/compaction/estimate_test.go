package compaction

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEstimateTokens_EmptyBody(t *testing.T) {
	result := EstimateTokens(map[string]any{})
	assert.Equal(t, 0, result)
}

func TestEstimateTokens_NilBody(t *testing.T) {
	result := EstimateTokens(nil)
	assert.Equal(t, 0, result)
}

func TestEstimateTokens_SimpleTextMessage(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "Hello, how are you?",
			},
		},
	}
	result := EstimateTokens(body)
	// "Hello, how are you?" = 21 chars → ~6 tokens + 1 role token = ~7
	assert.Greater(t, result, 0)
	assert.Less(t, result, 20) // should be well under 20
}

func TestEstimateTokens_ArrayContentBlocks(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Hello world"},
					map[string]any{
						"type": "tool_result",
						"tool_use_id": "toolu_123",
						"content": "Result text here",
					},
				},
			},
		},
	}
	result := EstimateTokens(body)
	assert.Greater(t, result, 0)
}

func TestEstimateTokens_ImageBlock(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Describe this"},
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "image/jpeg",
							"data":       "verylongbase64stringthatwouldbehugeinreality",
						},
					},
				},
			},
		},
	}
	result := EstimateTokens(body)
	// Should count ImageTokenEstimate (1000) for the image, not the full base64
	assert.GreaterOrEqual(t, result, ImageTokenEstimate)
	// Should be well under what raw base64 would give (base64 string is 49 chars → 13 tokens)
	assert.Less(t, result, 1200)
}

func TestEstimateTokens_DocumentBlock(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "Read this"},
					map[string]any{
						"type": "document",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "application/pdf",
							"data":       "verylongbase64data",
						},
					},
				},
			},
		},
	}
	result := EstimateTokens(body)
	assert.GreaterOrEqual(t, result, DocumentTokenEstimate)
}

func TestEstimateTokens_SystemPrompt(t *testing.T) {
	body := map[string]any{
		"system": "You are a helpful assistant that does things.",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}
	result := EstimateTokens(body)
	assert.Greater(t, result, 5)
}

func TestEstimateTokens_SystemPromptArray(t *testing.T) {
	body := map[string]any{
		"system": []any{
			map[string]any{"type": "text", "text": "You are helpful."},
			map[string]any{"type": "text", "text": "Be concise."},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
	}
	result := EstimateTokens(body)
	assert.Greater(t, result, 5)
}

func TestEstimateTokens_ToolsDefinitions(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"tools": []any{
			map[string]any{
				"name":         "get_weather",
				"description":  "Get the current weather for a location",
				"input_schema": map[string]any{"type": "object"},
			},
		},
	}
	result := EstimateTokens(body)
	assert.Greater(t, result, 10)
}

func TestEstimateTokens_ReasoningContent(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role":              "assistant",
				"content":           "The answer is 42.",
				"reasoning_content": "Let me think about this carefully...",
			},
		},
	}
	result := EstimateTokens(body)
	assert.Greater(t, result, 5)
}

func TestEstimateTokens_ToolUseBlock(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "text", "text": "Let me check."},
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
	result := EstimateTokens(body)
	assert.Greater(t, result, 3)
}

func TestEstimateTokens_MultipleMessages(t *testing.T) {
	messages := make([]any, 20)
	for i := 0; i < 20; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages[i] = map[string]any{
			"role":    role,
			"content": "This is a message with some content.",
		}
	}
	body := map[string]any{
		"messages": messages,
	}
	result := EstimateTokens(body)
	// 20 messages × ~9 tokens each ≈ 180 tokens
	assert.Greater(t, result, 100)
	assert.Less(t, result, 500)
}

func TestEstimateTokens_NonArrayMessages(t *testing.T) {
	body := map[string]any{
		"messages": "not an array",
	}
	result := EstimateTokens(body)
	assert.Equal(t, 0, result)
}

func TestEstimateTokens_NonMapMessageElement(t *testing.T) {
	body := map[string]any{
		"messages": []any{"not a map"},
	}
	result := EstimateTokens(body)
	// Should estimate from the JSON of the string itself
	assert.Greater(t, result, 0)
}

func TestCharToTokens(t *testing.T) {
	assert.Equal(t, 0, charToTokens(0))
	assert.Equal(t, 1, charToTokens(1))
	assert.Equal(t, 2, charToTokens(4))
	assert.Equal(t, 2, charToTokens(5))
	assert.Equal(t, 4, charToTokens(12))
}

func TestEstimateValueTokens(t *testing.T) {
	assert.Equal(t, 0, estimateValueTokens(nil))

	result := estimateValueTokens(map[string]any{"key": "value"})
	assert.Greater(t, result, 0)

	var v any
	result = estimateValueTokens(v)
	assert.Equal(t, 0, result)
}

func TestEstimateValueTokens_Unmarshallable(t *testing.T) {
	// Channels cannot be marshaled to JSON
	result := estimateValueTokens(make(chan int))
	assert.Equal(t, 0, result)
}

func TestEstimateContentTokens_UnknownType(t *testing.T) {
	// Integer content — falls through to estimateValueTokens
	result := estimateContentTokens(42)
	assert.Greater(t, result, 0)
}

func TestEstimateContentTokens_ToolResultWithArrayContent(t *testing.T) {
	content := []any{
		map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": []any{
			map[string]any{"type": "text", "text": "Part 1"},
			map[string]any{"type": "text", "text": "Part 2"},
		}},
	}
	body := map[string]any{"messages": []any{
		map[string]any{"role": "user", "content": content},
	}}
	result := EstimateTokens(body)
	assert.Greater(t, result, 2)
}

// Benchmark to ensure estimation is fast enough for per-request use.
func BenchmarkEstimateTokens_SmallRequest(b *testing.B) {
	body := map[string]any{
		"system": "You are a helpful assistant.",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hello"},
			map[string]any{"role": "assistant", "content": "Hi there!"},
		},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EstimateTokens(body)
	}
}

func BenchmarkEstimateTokens_LargeRequest(b *testing.B) {
	messages := make([]any, 50)
	for i := 0; i < 50; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		// Simulate a realistic message with content blocks
		messages[i] = map[string]any{
			"role": role,
			"content": []any{
				map[string]any{"type": "text", "text": "This is a longer message with some substantive content about programming and software development."},
			},
		}
	}
	body := map[string]any{
		"system":   "You are a helpful coding assistant.",
		"messages": messages,
		"tools": []any{
			map[string]any{
				"name":         "read_file",
				"description":  "Read a file from disk",
				"input_schema": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EstimateTokens(body)
	}
}

// Ensure the estimate function doesn't panic on unexpected types.
func TestEstimateTokens_FuzzyInput(t *testing.T) {
	// Various odd but valid JSON structures
	cases := []map[string]any{
		{"messages": []any{}},
		{"messages": []any{nil}},
		{"messages": []any{map[string]any{}}},
		{"messages": []any{map[string]any{"content": nil}}},
		{"messages": []any{map[string]any{"content": []any{}}}},
		{"messages": []any{map[string]any{"content": []any{nil}}}},
		{"system": 42},
		{"system": []any{42}},
	}

	for _, body := range cases {
		result := EstimateTokens(body)
		// Just ensure no panic and result is non-negative
		assert.GreaterOrEqual(t, result, 0, "body: %+v", body)
	}
}

// Verify JSON marshal/unmarshal roundtrip for content estimation.
func TestEstimateTokens_JsonRoundtrip(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": string(make([]byte, 1000))}, // 1000 null chars
				},
			},
		},
	}
	result := EstimateTokens(body)
	// 1000 chars / 4 = 250 tokens + overhead
	assert.Greater(t, result, 200)
}

// Verify the function is exported correctly.
func TestEstimateTokens_Exported(t *testing.T) {
	// Confirm the function signature matches what we expect
	var _ func(map[string]any) int = EstimateTokens
}

// Helper to create a body with N messages for testing.
func makeBodyWithNMessages(n int) map[string]any {
	messages := make([]any, n)
	for i := 0; i < n; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages[i] = map[string]any{
			"role":    role,
			"content": "A message with enough content to be realistic.",
		}
	}
	return map[string]any{"messages": messages}
}

func TestEstimateTokens_ScalesWithMessageCount(t *testing.T) {
	small := EstimateTokens(makeBodyWithNMessages(5))
	large := EstimateTokens(makeBodyWithNMessages(50))
	// Token count should scale roughly linearly
	assert.Greater(t, large, small*5)
}
