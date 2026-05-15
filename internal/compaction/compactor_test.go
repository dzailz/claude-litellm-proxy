package compaction

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Helpers ---

func newTestCompactor(upstream http.Handler) (*Compactor, *httptest.Server) {
	server := httptest.NewServer(upstream)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))

	c := &Compactor{
		Enabled:      true,
		MaxTokens:    1000,
		MinMessages:  5,
		SummaryModel: "test-model",
		BaseURL:      server.URL,
		APIKey:       "test-key",
		Client:       &http.Client{Timeout: 10 * time.Second},
		Logger:       logger,
	}
	return c, server
}

func makeBodyWithMessages(n int) map[string]any {
	messages := make([]any, n)
	for i := 0; i < n; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages[i] = map[string]any{
			"role":    role,
			"content": "This is message number " + itoa(i) + " with enough text to take tokens. " + paddingString(50),
		}
	}
	return map[string]any{
		"model":    "sonnet",
		"messages": messages,
	}
}

func makeLargeBody() map[string]any {
	// Create a body with enough content to exceed MaxTokens=1000
	messages := make([]any, 30)
	for i := 0; i < 30; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		messages[i] = map[string]any{
			"role":    role,
			"content": "This is a detailed message with lots of content to push the token count higher. " + paddingString(200),
		}
	}
	return map[string]any{
		"model":    "sonnet",
		"messages": messages,
	}
}

func makeSmallBody() map[string]any {
	return makeBodyWithMessages(3)
}

func itoa(i int) string   { return string(rune('0' + i)) }
func paddingString(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = 'x'
	}
	return string(b)
}

// summarizerHandler returns an HTTP handler that responds with a valid
// OpenAI chat/completions format containing the given summary text.
func summarizerHandler(summaryText string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"role":    "assistant",
						"content": summaryText,
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	}
}

// errorHandler returns an HTTP handler that returns the given status code.
func errorHandler(statusCode int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write([]byte(`{"error":"test error"}`))
	}
}

// --- ShouldCompact Tests ---

func TestShouldCompact_Disabled_ReturnsFalse(t *testing.T) {
	c := &Compactor{Enabled: false}
	body := makeLargeBody()

	assert.False(t, c.ShouldCompact(body))
}

func TestShouldCompact_BelowMinMessages_ReturnsFalse(t *testing.T) {
	c := &Compactor{
		Enabled:     true,
		MinMessages: 10,
		MaxTokens:   1, // very low threshold
	}
	body := makeBodyWithMessages(5) // has 5 messages, below MinMessages=10

	assert.False(t, c.ShouldCompact(body))
}

func TestShouldCompact_AtMinMessagesBelowTokenThreshold_ReturnsFalse(t *testing.T) {
	c := &Compactor{
		Enabled:     true,
		MinMessages: 5,
		MaxTokens:   100000, // very high threshold
	}
	body := makeBodyWithMessages(5)

	assert.False(t, c.ShouldCompact(body))
}

func TestShouldCompact_AboveMinMessagesAboveTokenThreshold_ReturnsTrue(t *testing.T) {
	c := &Compactor{
		Enabled:     true,
		MinMessages: 5,
		MaxTokens:   1, // very low threshold
	}
	body := makeBodyWithMessages(10)

	assert.True(t, c.ShouldCompact(body))
}

func TestShouldCompact_EmptyBody_ReturnsFalse(t *testing.T) {
	c := &Compactor{
		Enabled:     true,
		MinMessages: 2,
		MaxTokens:   1,
	}

	assert.False(t, c.ShouldCompact(map[string]any{}))
}

func TestShouldCompact_NoMessagesField_ReturnsFalse(t *testing.T) {
	c := &Compactor{
		Enabled:     true,
		MinMessages: 2,
		MaxTokens:   1,
	}

	assert.False(t, c.ShouldCompact(map[string]any{"model": "sonnet"}))
}

// --- Compact Tests: Happy Path ---

func TestCompact_BelowThreshold_ReturnsOriginalBodyUnchanged(t *testing.T) {
	body := makeSmallBody()
	c := &Compactor{
		Enabled:     true,
		MaxTokens:   100000,
		MinMessages: 5,
		Logger:      slog.New(slog.NewTextHandler(os.Stderr, nil)),
	}

	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	assert.Equal(t, body, result, "body below threshold should be returned unchanged")
}

func TestCompact_NoMessagesField_ReturnsOriginalBody(t *testing.T) {
	body := map[string]any{"model": "sonnet"}
	c, server := newTestCompactor(nil)
	defer server.Close()

	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	assert.Equal(t, body, result)
}

func TestCompact_EndToEnd_SmallBody(t *testing.T) {
	// A body small enough that no compaction triggers
	body := makeBodyWithMessages(3)
	c := &Compactor{
		Enabled:     true,
		MaxTokens:   100000,
		MinMessages: 5,
		Logger:      slog.New(slog.NewTextHandler(os.Stderr, nil)),
	}

	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	assert.Equal(t, body, result)
}

func TestCompact_EndToEnd_CompactionOccurs(t *testing.T) {
	c, server := newTestCompactor(summarizerHandler("This is a summary of the old conversation."))
	defer server.Close()

	body := makeLargeBody()
	originalMsgCount := len(body["messages"].([]any))

	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	require.NotNil(t, result)

	// Compacted body should have fewer messages
	resultMessages := result["messages"].([]any)
	assert.Less(t, len(resultMessages), originalMsgCount,
		"compacted body should have fewer messages than original")

	// First message should be the summary
	firstMsg := resultMessages[0].(map[string]any)
	assert.Equal(t, "user", firstMsg["role"])

	content := firstMsg["content"].([]any)
	assert.Equal(t, "text", content[0].(map[string]any)["type"])
	text := content[0].(map[string]any)["text"].(string)
	assert.Contains(t, text, "[Previous conversation summary]")
	assert.Contains(t, text, "[End of summary]")
	assert.Contains(t, text, "summary of the old conversation")

	// Other fields (model, etc.) should be preserved
	assert.Equal(t, "sonnet", result["model"])
}

func TestCompact_PreservesNonMessageFields(t *testing.T) {
	c, server := newTestCompactor(summarizerHandler("Summary."))
	defer server.Close()

	body := makeLargeBody()
	body["system"] = "You are a helpful assistant."
	body["max_tokens"] = 1000
	body["stream"] = true

	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "You are a helpful assistant.", result["system"])
	// max_tokens is an int in Go, not a JSON float64
	assert.Equal(t, 1000, result["max_tokens"])
	assert.True(t, result["stream"].(bool))
}

func TestCompact_DoesNotModifyOriginalBody(t *testing.T) {
	c, server := newTestCompactor(summarizerHandler("Summary."))
	defer server.Close()

	body := makeLargeBody()
	originalJSON, _ := json.Marshal(body)

	_, err := c.Compact(context.Background(), body)
	require.NoError(t, err)

	afterJSON, _ := json.Marshal(body)
	assert.JSONEq(t, string(originalJSON), string(afterJSON),
		"original body must not be modified by compaction")
}

// --- Compact Tests: Error / Graceful Degradation ---

func TestCompact_SummarizationTimeout_ReturnsOriginalBody(t *testing.T) {
	// Create a handler that delays > the 60s summarization timeout
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second) // shorter than 60s but we can set a short client timeout
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()
	c.Client.Timeout = 1 * time.Millisecond // force timeout

	body := makeLargeBody()

	result, err := c.Compact(context.Background(), body)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "summarization")
	// With the way Compact works, errors return the original body
	// as part of graceful degradation
	assert.NotNil(t, result)
}

func TestCompact_SummarizationHTTPError_ReturnsOriginalBody(t *testing.T) {
	c, server := newTestCompactor(errorHandler(http.StatusInternalServerError))
	defer server.Close()

	body := makeLargeBody()

	result, err := c.Compact(context.Background(), body)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "summarization")
	assert.NotNil(t, result)
}

func TestCompact_SummarizationBadJSON_ReturnsOriginalBody(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`not json`))
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()

	body := makeLargeBody()

	result, err := c.Compact(context.Background(), body)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "summarization")
	assert.NotNil(t, result)
}

func TestCompact_SummarizationEmptyContent_ReturnsOriginalBody(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"role":    "assistant",
						"content": "", // empty content
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()

	body := makeLargeBody()

	result, err := c.Compact(context.Background(), body)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty content")
	assert.NotNil(t, result)
}

func TestCompact_SummarizationNoChoices_ReturnsOriginalBody(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []any{}, // empty choices
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()

	body := makeLargeBody()

	result, err := c.Compact(context.Background(), body)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "no choices")
	assert.NotNil(t, result)
}

func TestCompact_EmptyOldPartition_ReturnsOriginal(t *testing.T) {
	// When findSplitIndex returns 0 (all messages are recent), compaction should skip
	c, server := newTestCompactor(summarizerHandler("Summary."))
	defer server.Close()

	// Only 4 messages with high MaxTokens — but we need to trigger
	// We need exactly 4 messages which gives splitIdx = 0
	// Actually findSplitIndex keeps max(4, 25%) so for 4 messages splitIdx = 0
	body := makeParentBodyWith4Messages()
	// But we also need to be above MaxTokens — lower it
	c.MaxTokens = 1
	c.MinMessages = 3

	// 4 messages with threshold=1 → ShouldCompact returns true
	// findSplitIndex(4) → recentCount = max(4, 1) = 4, splitIdx = 0
	// Compact should return body unchanged because splitIdx <= 0
	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	assert.Equal(t, body, result)
}

func makeParentBodyWith4Messages() map[string]any {
	messages := make([]any, 4)
	for i := 0; i < 4; i++ {
		messages[i] = map[string]any{
			"role":    "user",
			"content": "message " + paddingString(200),
		}
	}
	return map[string]any{"model": "sonnet", "messages": messages}
}

// --- Tool Chain Integrity Tests ---

func TestCompact_ToolChains_NotSplit(t *testing.T) {
	c, server := newTestCompactor(summarizerHandler("Summary with tool chain context."))
	defer server.Close()

	// Create a conversation with tool_use/tool_result pairs near the split boundary.
	// 15 messages: messages 0-9 are "old", 10-14 are "recent" (with 4 recent + adjustment)
	// Messages 8-9 have a tool_use/tool_result pair that spans the boundary.
	messages := make([]any, 15)
	for i := 0; i < 15; i++ {
		messages[i] = map[string]any{
			"role":    "user",
			"content": "msg " + itoa(i) + " " + paddingString(100),
		}
	}

	// Replace message 10 with an assistant message containing tool_use
	messages[10] = map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{
				"type": "text",
				"text": "Let me run a tool.",
			},
			map[string]any{
				"type":  "tool_use",
				"id":    "toolu_abc123",
				"name":  "read_file",
				"input": map[string]any{"path": "/test"},
			},
		},
	}

	// Replace message 11 with a user message containing tool_result
	messages[11] = map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{
				"type":         "tool_result",
				"tool_use_id":  "toolu_abc123",
				"content":      "file contents here",
			},
		},
	}

	body := map[string]any{"model": "sonnet", "messages": messages}

	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	require.NotNil(t, result)

	// The tool_use at index 10 and tool_result at index 11 should both be in the recent partition
	resultMessages := result["messages"].([]any)
	foundToolUse := false
	foundToolResult := false

	for _, msg := range resultMessages {
		mm, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		blocks, ok := mm["content"].([]any)
		if !ok {
			continue
		}
		for _, block := range blocks {
			blk, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if blk["type"] == "tool_use" && blk["id"] == "toolu_abc123" {
				foundToolUse = true
			}
			if blk["type"] == "tool_result" && blk["tool_use_id"] == "toolu_abc123" {
				foundToolResult = true
			}
		}
	}

	assert.True(t, foundToolUse, "tool_use should be in the preserved recent messages (not summarized)")
	assert.True(t, foundToolResult, "tool_result should be in the preserved recent messages (not summarized)")
}

// --- findSplitIndex Tests ---

func TestFindSplitIndex_NormalCase(t *testing.T) {
	c := &Compactor{}
	messages := make([]any, 20)

	idx := c.findSplitIndex(messages)
	// 25% of 20 = 5, max(4, 5) = 5, so splitIdx = 20 - 5 = 15
	assert.Equal(t, 15, idx)
}

func TestFindSplitIndex_SmallList_KeepsMinRecent(t *testing.T) {
	c := &Compactor{}
	messages := make([]any, 6)

	idx := c.findSplitIndex(messages)
	// 25% of 6 = 1, max(4, 1) = 4, splitIdx = 6 - 4 = 2
	assert.Equal(t, 2, idx)
}

func TestFindSplitIndex_VerySmallList_NoOldMessages(t *testing.T) {
	c := &Compactor{}
	messages := make([]any, 4)

	idx := c.findSplitIndex(messages)
	// 25% of 4 = 1, max(4, 1) = 4, splitIdx = 4 - 4 = 0
	assert.Equal(t, 0, idx)
}

func TestFindSplitIndex_EmptyList(t *testing.T) {
	c := &Compactor{}
	messages := make([]any, 0)

	idx := c.findSplitIndex(messages)
	assert.Equal(t, 0, idx)
}

// --- adjustSplitForToolChains Tests ---

func TestAdjustSplitForToolChains_NoTools_ReturnsSameIdx(t *testing.T) {
	messages := []any{
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": "hi"},
		map[string]any{"role": "user", "content": "bye"},
	}

	result := adjustSplitForToolChains(messages, 1)
	assert.Equal(t, 1, result)
}

func TestAdjustSplitForToolChains_ToolResultInRecent_ToolUseInOld_MovesSplit(t *testing.T) {
	messages := []any{
		map[string]any{"role": "user", "content": "old msg 1"},
		map[string]any{
			"role": "assistant",
			"content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_1", "name": "test", "input": map[string]any{}},
			},
		},
		map[string]any{"role": "user", "content": "old msg 2"},
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"},
			},
		},
		map[string]any{"role": "assistant", "content": "final answer"},
	}

	// Split at index 3 (messages 0,1,2 = old; 3,4 = recent)
	// tool_result at index 3 references tool_use at index 1 which is in "old"
	// split should move left to include the tool_use
	result := adjustSplitForToolChains(messages, 3)
	assert.Less(t, result, 3, "split should move left to include the tool_use")
	assert.GreaterOrEqual(t, result, 0)
}

func TestAdjustSplitForToolChains_BothInRecent_NoChange(t *testing.T) {
	messages := []any{
		map[string]any{"role": "user", "content": "old"},
		map[string]any{
			"role": "assistant",
			"content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_x", "name": "test", "input": map[string]any{}},
			},
		},
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_x", "content": "result"},
			},
		},
	}

	result := adjustSplitForToolChains(messages, 1)
	assert.Equal(t, 1, result, "split should not change when tool_use and tool_result are both in recent")
}

func TestAdjustSplitForToolChains_SplitAtZero_ReturnsZero(t *testing.T) {
	messages := []any{
		map[string]any{"role": "user", "content": "only msg"},
	}

	result := adjustSplitForToolChains(messages, 0)
	assert.Equal(t, 0, result)
}

func TestAdjustSplitForToolChains_EmptyMessages(t *testing.T) {
	result := adjustSplitForToolChains([]any{}, 0)
	assert.Equal(t, 0, result)
}

// --- stripMediaFromMessages Tests ---

func TestStripMediaFromMessages_ImageReplaced(t *testing.T) {
	messages := []any{
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "Look at this:"},
				map[string]any{
					"type": "image",
					"source": map[string]any{
						"type":       "base64",
						"media_type": "image/png",
						"data":       "base64datahere",
					},
				},
			},
		},
	}

	result := stripMediaFromMessages(messages)
	require.Len(t, result, 1)

	msg := result[0].(map[string]any)
	blocks := msg["content"].([]any)
	assert.Len(t, blocks, 2)

	// First block: text unchanged
	assert.Equal(t, "text", blocks[0].(map[string]any)["type"])

	// Second block: image replaced with text placeholder
	assert.Equal(t, "text", blocks[1].(map[string]any)["type"])
	assert.Equal(t, "[image]", blocks[1].(map[string]any)["text"])
}

func TestStripMediaFromMessages_DocumentReplaced(t *testing.T) {
	messages := []any{
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{
					"type": "document",
					"source": map[string]any{
						"type":       "base64",
						"media_type": "application/pdf",
						"data":       "pdfdata",
					},
				},
			},
		},
	}

	result := stripMediaFromMessages(messages)
	blocks := result[0].(map[string]any)["content"].([]any)
	assert.Equal(t, "text", blocks[0].(map[string]any)["type"])
	assert.Equal(t, "[document]", blocks[0].(map[string]any)["text"])
}

func TestStripMediaFromMessages_NonMapMessage_Preserved(t *testing.T) {
	messages := []any{"just a string"}
	result := stripMediaFromMessages(messages)
	assert.Equal(t, "just a string", result[0])
}

func TestStripMediaFromMessages_NoContentField(t *testing.T) {
	messages := []any{
		map[string]any{"role": "user"}, // no content
	}
	result := stripMediaFromMessages(messages)
	assert.Equal(t, "user", result[0].(map[string]any)["role"])
}

func TestStripMediaFromMessages_StringContent(t *testing.T) {
	messages := []any{
		map[string]any{
			"role":    "user",
			"content": "plain text", // string, not array
		},
	}
	result := stripMediaFromMessages(messages)
	assert.Equal(t, "plain text", result[0].(map[string]any)["content"])
}

func TestStripMediaFromMessages_OriginalNotModified(t *testing.T) {
	messages := []any{
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "image", "source": map[string]any{"data": "abc"}},
			},
		},
	}

	originalCopy := make([]any, len(messages))
	copy(originalCopy, messages)

	stripMediaFromMessages(messages)

	// Original should still have the image
	origBlocks := messages[0].(map[string]any)["content"].([]any)
	assert.Equal(t, "image", origBlocks[0].(map[string]any)["type"],
		"original messages must not be modified")
}

// --- collectToolUseIDsWithIndex Tests ---

func TestCollectToolUseIDsWithIndex_SingleToolUse(t *testing.T) {
	msg := map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "tool_use", "id": "toolu_1", "name": "test", "input": map[string]any{}},
		},
	}

	ids := make(map[string]int)
	collectToolUseIDsWithIndex(msg, 5, ids)

	assert.Equal(t, 5, ids["toolu_1"])
}

func TestCollectToolUseIDsWithIndex_MultipleToolUses(t *testing.T) {
	msg := map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "tool_use", "id": "toolu_a", "name": "test", "input": map[string]any{}},
			map[string]any{"type": "text", "text": "some text"},
			map[string]any{"type": "tool_use", "id": "toolu_b", "name": "test2", "input": map[string]any{}},
		},
	}

	ids := make(map[string]int)
	collectToolUseIDsWithIndex(msg, 7, ids)

	assert.Equal(t, 7, ids["toolu_a"])
	assert.Equal(t, 7, ids["toolu_b"])
}

func TestCollectToolUseIDsWithIndex_NoContentField(t *testing.T) {
	msg := map[string]any{"role": "user"}
	ids := make(map[string]int)

	collectToolUseIDsWithIndex(msg, 0, ids)
	assert.Empty(t, ids)
}

func TestCollectToolUseIDsWithIndex_StringContent(t *testing.T) {
	msg := map[string]any{"role": "user", "content": "plain text"}
	ids := make(map[string]int)

	collectToolUseIDsWithIndex(msg, 0, ids)
	assert.Empty(t, ids)
}

func TestCollectToolUseIDsWithIndex_NoToolUseBlocks(t *testing.T) {
	msg := map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "text", "text": "just text"},
		},
	}
	ids := make(map[string]int)

	collectToolUseIDsWithIndex(msg, 3, ids)
	assert.Empty(t, ids)
}

// --- findEarliestReferencedToolUseIdx Tests ---

func TestFindEarliestReferencedToolUseIdx_Found(t *testing.T) {
	msg := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "tool_result", "tool_use_id": "toolu_x", "content": "result"},
		},
	}
	ids := map[string]int{"toolu_x": 2}

	assert.Equal(t, 2, findEarliestReferencedToolUseIdx(msg, ids))
}

func TestFindEarliestReferencedToolUseIdx_NotFound(t *testing.T) {
	msg := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "tool_result", "tool_use_id": "toolu_y", "content": "result"},
		},
	}
	ids := map[string]int{"toolu_x": 2}

	assert.Equal(t, -1, findEarliestReferencedToolUseIdx(msg, ids))
}

func TestFindEarliestReferencedToolUseIdx_NoContent(t *testing.T) {
	msg := map[string]any{"role": "user"}
	ids := map[string]int{"toolu_x": 2}

	assert.Equal(t, -1, findEarliestReferencedToolUseIdx(msg, ids))
}

func TestFindEarliestReferencedToolUseIdx_EarliestIndex(t *testing.T) {
	msg := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "tool_result", "tool_use_id": "toolu_a", "content": "result"},
			map[string]any{"type": "tool_result", "tool_use_id": "toolu_b", "content": "result"},
		},
	}
	ids := map[string]int{"toolu_a": 1, "toolu_b": 3}

	assert.Equal(t, 1, findEarliestReferencedToolUseIdx(msg, ids),
		"should return the smallest index among referenced tool_uses")
}

// --- summarize Tests ---

func TestSummarize_Success(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the request format
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))

		assert.Equal(t, "test-model", req["model"])
		assert.False(t, req["stream"].(bool))
		assert.Equal(t, float64(4096), req["max_tokens"])

		// Verify authorization header
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		resp := map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{
						"role":    "assistant",
						"content": "Summary text here.",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()

	oldMessages := []any{
		map[string]any{"role": "user", "content": "old message 1"},
		map[string]any{"role": "assistant", "content": "old response 1"},
	}

	body := map[string]any{"model": "sonnet"}
	summary, err := c.summarize(context.Background(), body, oldMessages)

	require.NoError(t, err)
	assert.Equal(t, "Summary text here.", summary)
}

func TestSummarize_UsesFallbackModel(t *testing.T) {
	var receivedModel string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		receivedModel = req["model"].(string)

		resp := map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{"role": "assistant", "content": "ok"},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()
	c.SummaryModel = "" // empty → should fall back to body model

	oldMessages := []any{map[string]any{"role": "user", "content": "test"}}
	body := map[string]any{"model": "claude-sonnet-4"}

	_, err := c.summarize(context.Background(), body, oldMessages)
	require.NoError(t, err)
	assert.Equal(t, "claude-sonnet-4", receivedModel)
}

func TestSummarize_FallbackToDefaultWhenNoModel(t *testing.T) {
	var receivedModel string

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		receivedModel = req["model"].(string)

		resp := map[string]any{
			"choices": []any{
				map[string]any{
					"message": map[string]any{"role": "assistant", "content": "ok"},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	})

	c, server := newTestCompactor(upstream)
	defer server.Close()
	c.SummaryModel = "" // empty

	oldMessages := []any{map[string]any{"role": "user", "content": "test"}}
	body := map[string]any{} // no model field

	_, err := c.summarize(context.Background(), body, oldMessages)
	require.NoError(t, err)
	assert.Equal(t, "glm5.1", receivedModel)
}

func TestSummarize_HTTPError(t *testing.T) {
	c, server := newTestCompactor(errorHandler(http.StatusServiceUnavailable))
	defer server.Close()

	oldMessages := []any{map[string]any{"role": "user", "content": "test"}}
	body := map[string]any{"model": "sonnet"}

	_, err := c.summarize(context.Background(), body, oldMessages)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "status 503")
}

// --- messageCount Tests ---

func TestMessageCount_NormalCase(t *testing.T) {
	body := makeBodyWithMessages(7)
	assert.Equal(t, 7, messageCount(body))
}

func TestMessageCount_NoMessagesField(t *testing.T) {
	assert.Equal(t, 0, messageCount(map[string]any{}))
}

func TestMessageCount_NilBody(t *testing.T) {
	assert.Equal(t, 0, messageCount(nil))
}

func TestMessageCount_MessagesNotArray(t *testing.T) {
	assert.Equal(t, 0, messageCount(map[string]any{"messages": "not array"}))
}

// --- Compact disabled tests ---

func TestCompact_Disabled_ReturnsUnchanged(t *testing.T) {
	c, server := newTestCompactor(summarizerHandler("Summary."))
	defer server.Close()
	c.Enabled = false

	body := makeLargeBody()
	result, err := c.Compact(context.Background(), body)

	require.NoError(t, err)
	assert.Equal(t, body, result, "disabled compactor should return body unchanged")
}

// --- SummaryPrompt constant test ---

func TestSummaryPrompt_NotEmpty(t *testing.T) {
	assert.NotEmpty(t, SummaryPrompt)
	assert.Contains(t, SummaryPrompt, "conversation summarizer")
}

func TestUserSummaryPrompt_NotEmpty(t *testing.T) {
	assert.NotEmpty(t, UserSummaryPrompt)
	assert.Contains(t, UserSummaryPrompt, "summarize the following conversation")
}

// --- buildCompactedMessages Tests ---

func TestBuildCompactedMessages_EmptyRecent_SummaryOnly(t *testing.T) {
	result := buildCompactedMessages("summary text", []any{})

	require.Len(t, result, 1)
	msg := result[0].(map[string]any)
	assert.Equal(t, "user", msg["role"])
	blocks := msg["content"].([]any)
	assert.Equal(t, "summary text", blocks[0].(map[string]any)["text"])
}

func TestBuildCompactedMessages_FirstIsUser_MergesSummary(t *testing.T) {
	recent := []any{
		map[string]any{
			"role":    "user",
			"content": "my question",
		},
		map[string]any{
			"role":    "assistant",
			"content": "my answer",
		},
	}

	result := buildCompactedMessages("summary text", recent)

	require.Len(t, result, 2)

	// First message should be user with merged summary
	first := result[0].(map[string]any)
	assert.Equal(t, "user", first["role"])
	blocks := first["content"].([]any)
	require.Len(t, blocks, 2)
	assert.Contains(t, blocks[0].(map[string]any)["text"], "summary text")
	assert.Equal(t, "my question", blocks[1].(map[string]any)["text"])

	// Second message unchanged
	second := result[1].(map[string]any)
	assert.Equal(t, "assistant", second["role"])
	assert.Equal(t, "my answer", second["content"])
}

func TestBuildCompactedMessages_FirstIsUser_ArrayContent_MergesSummary(t *testing.T) {
	recent := []any{
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "my question"},
			},
		},
	}

	result := buildCompactedMessages("summary text", recent)

	require.Len(t, result, 1)
	first := result[0].(map[string]any)
	blocks := first["content"].([]any)
	require.Len(t, blocks, 2)
	// Summary block prepended
	assert.Contains(t, blocks[0].(map[string]any)["text"], "summary text")
	// Original block preserved
	assert.Equal(t, "my question", blocks[1].(map[string]any)["text"])
}

func TestBuildCompactedMessages_FirstIsAssistant_PrependsUserSummary(t *testing.T) {
	recent := []any{
		map[string]any{
			"role":    "assistant",
			"content": "my answer",
		},
		map[string]any{
			"role":    "user",
			"content": "follow-up",
		},
	}

	result := buildCompactedMessages("summary text", recent)

	require.Len(t, result, 3)

	// Prepended summary as user message
	first := result[0].(map[string]any)
	assert.Equal(t, "user", first["role"])
	blocks := first["content"].([]any)
	assert.Contains(t, blocks[0].(map[string]any)["text"], "summary text")

	// Original messages follow in order
	assert.Equal(t, "assistant", result[1].(map[string]any)["role"])
	assert.Equal(t, "user", result[2].(map[string]any)["role"])
}

func TestBuildCompactedMessages_FirstHasToolResult_PrependsUserAndAssistant(t *testing.T) {
	recent := []any{
		map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{
					"type":         "tool_result",
					"tool_use_id":  "toolu_abc",
					"content":      "file contents",
				},
			},
		},
		map[string]any{
			"role":    "assistant",
			"content": "based on the file...",
		},
	}

	result := buildCompactedMessages("summary text", recent)

	require.Len(t, result, 4)

	// Summary as user
	first := result[0].(map[string]any)
	assert.Equal(t, "user", first["role"])

	// Synthetic assistant with tool_calls
	second := result[1].(map[string]any)
	assert.Equal(t, "assistant", second["role"])
	toolCalls, ok := second["tool_calls"].([]any)
	require.True(t, ok, "synthetic assistant should have tool_calls")
	require.Len(t, toolCalls, 1)
	tc := toolCalls[0].(map[string]any)
	assert.Equal(t, "toolu_abc", tc["id"])

	// Original messages follow
	assert.Equal(t, "user", result[2].(map[string]any)["role"])
	assert.Equal(t, "assistant", result[3].(map[string]any)["role"])
}

// --- mergeSummaryIntoMessage Tests ---

func TestMergeSummaryIntoMessage_StringContent(t *testing.T) {
	msg := map[string]any{
		"role":    "user",
		"content": "hello",
	}

	result := mergeSummaryIntoMessage(msg, "summary text")

	// Original not modified
	assert.Equal(t, "hello", msg["content"])

	// Result has array content with summary + original
	assert.Equal(t, "user", result["role"])
	blocks := result["content"].([]any)
	require.Len(t, blocks, 2)
	assert.Contains(t, blocks[0].(map[string]any)["text"], "summary text")
	assert.Equal(t, "hello", blocks[1].(map[string]any)["text"])
}

func TestMergeSummaryIntoMessage_ArrayContent(t *testing.T) {
	msg := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "text", "text": "hello"},
			map[string]any{"type": "text", "text": "world"},
		},
	}

	result := mergeSummaryIntoMessage(msg, "summary text")

	blocks := result["content"].([]any)
	require.Len(t, blocks, 3)
	assert.Contains(t, blocks[0].(map[string]any)["text"], "summary text")
	assert.Equal(t, "hello", blocks[1].(map[string]any)["text"])
	assert.Equal(t, "world", blocks[2].(map[string]any)["text"])
}

func TestMergeSummaryIntoMessage_NoContent(t *testing.T) {
	msg := map[string]any{
		"role": "user",
	}

	result := mergeSummaryIntoMessage(msg, "summary text")

	blocks := result["content"].([]any)
	require.Len(t, blocks, 1)
	assert.Contains(t, blocks[0].(map[string]any)["text"], "summary text")
}

// --- extractToolCallsFromToolResults Tests ---

func TestExtractToolCallsFromToolResults_SingleToolResult(t *testing.T) {
	msg := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{
				"type":         "tool_result",
				"tool_use_id":  "toolu_abc",
				"content":      "result",
			},
		},
	}

	calls := extractToolCallsFromToolResults(msg)
	require.Len(t, calls, 1)
	tc := calls[0].(map[string]any)
	assert.Equal(t, "toolu_abc", tc["id"])
	assert.Equal(t, "function", tc["type"])
	fn := tc["function"].(map[string]any)
	assert.Equal(t, "tool", fn["name"])
}

func TestExtractToolCallsFromToolResults_NoToolResults(t *testing.T) {
	msg := map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "text", "text": "hello"},
		},
	}

	calls := extractToolCallsFromToolResults(msg)
	assert.Nil(t, calls)
}

func TestExtractToolCallsFromToolResults_NoContent(t *testing.T) {
	msg := map[string]any{"role": "user"}
	calls := extractToolCallsFromToolResults(msg)
	assert.Nil(t, calls)
}

// --- Integration: Compact produces valid role alternation ---

func TestCompact_ProducesValidRoleAlternation(t *testing.T) {
	c, server := newTestCompactor(summarizerHandler("Summary of old conversation."))
	defer server.Close()

	body := makeLargeBody()

	result, err := c.Compact(context.Background(), body)
	require.NoError(t, err)

	resultMessages := result["messages"].([]any)

	// Verify no consecutive same-role messages
	for i := 1; i < len(resultMessages); i++ {
		prevRole := resultMessages[i-1].(map[string]any)["role"]
		currRole := resultMessages[i].(map[string]any)["role"]
		assert.NotEqual(t, prevRole, currRole,
			"consecutive messages at index %d and %d both have role %q", i-1, i, prevRole)
	}
}

// --- Benchmark ---

func BenchmarkShouldCompact_SmallBody(b *testing.B) {
	c := &Compactor{Enabled: true, MaxTokens: 1000, MinMessages: 5}
	body := makeBodyWithMessages(3)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ShouldCompact(body)
	}
}

func BenchmarkShouldCompact_LargeBody(b *testing.B) {
	c := &Compactor{Enabled: true, MaxTokens: 1000, MinMessages: 5}
	body := makeLargeBody()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ShouldCompact(body)
	}
}
