package compaction

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// Compactor performs server-side transparent conversation compaction.
// When a request's estimated token count exceeds the configured threshold,
// older messages are summarized into a compact form to fit within the
// upstream model's context window.
//
// Compaction is designed to be transparent to the client (Claude Code):
// it modifies the messages array in-flight before forwarding upstream,
// and the client never knows compaction happened.
type Compactor struct {
	// Enabled controls whether compaction is active.
	Enabled bool

	// MaxTokens is the token threshold above which compaction triggers.
	// Default: 80000 (80% of a 100k context window).
	MaxTokens int

	// SummaryModel is the model identifier used for summarization requests.
	// If empty, the request's own model is used.
	SummaryModel string

	// MinMessages is the minimum number of messages required before compaction
	// is considered. Small conversations that exceed the token limit are
	// likely caused by large individual messages (images, documents) that
	// compaction cannot help with.
	MinMessages int

	// BaseURL is the upstream proxy endpoint for summarization requests
	// (OpenAI chat/completions format).
	BaseURL string

	// APIKey is the authentication key for the upstream proxy.
	APIKey string

	// Client is the HTTP client used for summarization requests.
	Client *http.Client

	// Logger is the structured logger for compaction events.
	Logger *slog.Logger
}

// SummaryPrompt is the system prompt used for the summarization model.
const SummaryPrompt = `You are a conversation summarizer. Your task is to create a concise but complete summary of the conversation so far. 

Preserve:
- Key decisions and conclusions
- Important file paths, function names, and code references
- Tool calls made and their results (especially errors or important outputs)
- The current task or goal
- Any constraints or requirements mentioned

Be concise but do not lose critical context. Format the summary as a structured narrative.`

// UserSummaryPrompt prefixes the conversation to be summarized.
const UserSummaryPrompt = "Please summarize the following conversation, preserving all important context, decisions, code references, and tool results:\n\n"

// ShouldCompact returns true if the request body exceeds the token threshold
// and has enough messages to benefit from compaction.
func (c *Compactor) ShouldCompact(body map[string]any) bool {
	if !c.Enabled {
		return false
	}

	messages := messageCount(body)
	if messages < c.MinMessages {
		return false
	}

	estimated := EstimateTokens(body)
	return estimated > c.MaxTokens
}

// Compact performs conversation compaction on the request body.
// It splits messages into "old" (to summarize) and "recent" (to preserve),
// calls the summarization model, and replaces old messages with the summary.
//
// If compaction fails for any reason, it returns the original body unchanged
// (graceful degradation). The caller should log the error and proceed.
func (c *Compactor) Compact(ctx context.Context, body map[string]any) (map[string]any, error) {
	estimatedTokens := EstimateTokens(body)
	msgCount := messageCount(body)

	c.Logger.Info("compaction check",
		"enabled", c.Enabled,
		"estimated_tokens", estimatedTokens,
		"max_tokens", c.MaxTokens,
		"message_count", msgCount,
		"min_messages", c.MinMessages,
		"should_compact", c.ShouldCompact(body),
	)

	if !c.ShouldCompact(body) {
		return body, nil
	}

	messagesRaw, ok := body["messages"].([]any)
	if !ok || len(messagesRaw) < c.MinMessages {
		return body, nil
	}

	// Split messages into old (to summarize) and recent (to preserve)
	splitIdx := c.findSplitIndex(messagesRaw)
	if splitIdx <= 0 {
		// Can't compact — all messages need to be preserved
		return body, nil
	}

	oldMessages := messagesRaw[:splitIdx]
	recentMessages := messagesRaw[splitIdx:]

	// Strip images and documents from old messages before summarization
	strippedOld := stripMediaFromMessages(oldMessages)

	// Build the summarization request
	summaryText, err := c.summarize(ctx, body, strippedOld)
	if err != nil {
		return body, fmt.Errorf("summarization failed: %w", err)
	}

	// Build the summary text block used across all insertion strategies.
	summaryBlockText := fmt.Sprintf("[Previous conversation summary]\n%s\n[End of summary]", summaryText)

	// Build the compacted messages array. We must avoid consecutive same-role
	// messages (e.g., two "user" messages in a row), which violates both
	// Anthropic and OpenAI conversation format requirements.
	compactedMessages := buildCompactedMessages(summaryBlockText, recentMessages)

	// Log which insertion strategy was used (helps debug role alternation issues)
	firstRecentRole, _ := recentMessages[0].(map[string]any)["role"].(string)
	hasToolResult := firstRecentRole == "tool" || containsToolResult(recentMessages[0].(map[string]any))
	strategy := "prepend_summary" // case 2: assistant first
	switch {
	case hasToolResult:
		strategy = "synthetic_assistant" // case 3: tool_result first
	case firstRecentRole == "user":
		strategy = "merge_into_user" // case 1: user first
	}
	c.Logger.Debug("compaction strategy",
		"strategy", strategy,
		"first_recent_role", firstRecentRole,
		"first_recent_has_tool_result", hasToolResult,
	)

	// Create a new body with compacted messages (immutable — don't modify original)
	newBody := make(map[string]any, len(body))
	for k, v := range body {
		newBody[k] = v
	}
	newBody["messages"] = compactedMessages

	originalTokens := EstimateTokens(body)
	compactedTokens := EstimateTokens(newBody)

	c.Logger.Info("compaction completed",
		"original_messages", len(messagesRaw),
		"compacted_messages", len(compactedMessages),
		"old_messages_summarized", len(oldMessages),
		"recent_messages_preserved", len(recentMessages),
		"original_tokens_estimated", originalTokens,
		"compacted_tokens_estimated", compactedTokens,
		"token_reduction", originalTokens-compactedTokens,
		"role_sequence", roleSequence(compactedMessages),
	)

	return newBody, nil
}

// findSplitIndex determines where to split the messages array between
// "old" (to summarize) and "recent" (to preserve). It keeps the last
// max(4, 25% of messages) messages intact, adjusted to preserve
// tool_use/tool_result pairs.
func (c *Compactor) findSplitIndex(messages []any) int {
	n := len(messages)
	recentCount := n / 4
	if recentCount < 4 {
		recentCount = 4
	}
	if recentCount > n {
		recentCount = n
	}

	// Start from the boundary and adjust to preserve tool chains
	splitIdx := n - recentCount

	// Ensure we don't split a tool_use / tool_result pair
	// Walk backward from splitIdx; if the message at splitIdx is a
	// tool_result whose tool_use is in the "old" partition, move
	// the split earlier to include the tool_use in "recent"
	splitIdx = adjustSplitForToolChains(messages, splitIdx)

	return splitIdx
}

// adjustSplitForToolChains ensures tool_use/tool_result pairs are not
// split across the old/recent boundary. If a tool_result in the "recent"
// partition references a tool_use in the "old" partition, the tool_use
// is moved into "recent" by adjusting the split index downward to the
// earliest referenced tool_use.
func adjustSplitForToolChains(messages []any, splitIdx int) int {
	if splitIdx <= 0 {
		return 0
	}

	// Collect all tool_use IDs in the "old" partition with their indices.
	oldToolUseIDs := make(map[string]int) // id → index
	for i := 0; i < splitIdx; i++ {
		if mm, ok := messages[i].(map[string]any); ok {
			collectToolUseIDsWithIndex(mm, i, oldToolUseIDs)
		}
	}

	// Find the earliest "old" tool_use referenced by a "recent" tool_result.
	minIdx := splitIdx
	for i := splitIdx; i < len(messages); i++ {
		if mm, ok := messages[i].(map[string]any); ok {
			if idx := findEarliestReferencedToolUseIdx(mm, oldToolUseIDs); idx >= 0 && idx < minIdx {
				minIdx = idx
			}
		}
	}

	return minIdx
}

// collectToolUseIDsWithIndex extracts all tool_use block IDs and their
// message index from a message's content.
func collectToolUseIDsWithIndex(msg map[string]any, msgIdx int, ids map[string]int) {
	content, ok := msg["content"]
	if !ok {
		return
	}

	blocks, ok := content.([]any)
	if !ok {
		return
	}

	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			continue
		}
		if blk["type"] == "tool_use" {
			if id, ok := blk["id"].(string); ok && id != "" {
				ids[id] = msgIdx
			}
		}
	}
}

// findEarliestReferencedToolUseIdx finds the smallest index of any
// old-partition tool_use referenced by a tool_result block in this message.
// Returns -1 if no old tool_use is referenced.
func findEarliestReferencedToolUseIdx(msg map[string]any, oldToolUseIDs map[string]int) int {
	content, ok := msg["content"]
	if !ok {
		return -1
	}

	blocks, ok := content.([]any)
	if !ok {
		return -1
	}

	bestIdx := -1
	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			continue
		}
		if blk["type"] == "tool_result" {
			if toolUseID, ok := blk["tool_use_id"].(string); ok {
				if idx, found := oldToolUseIDs[toolUseID]; found {
					if bestIdx < 0 || idx < bestIdx {
						bestIdx = idx
					}
				}
			}
		}
	}
	return bestIdx
}

// stripMediaFromMessages replaces images and documents with placeholder
// text in all messages, reducing the token count before summarization.
// Returns a new slice — does not modify the input.
func stripMediaFromMessages(messages []any) []any {
	result := make([]any, len(messages))
	for i, msg := range messages {
		mm, ok := msg.(map[string]any)
		if !ok {
			result[i] = msg
			continue
		}

		content, ok := mm["content"]
		if !ok {
			result[i] = msg
			continue
		}

		blocks, ok := content.([]any)
		if !ok {
			// String content — no media to strip
			result[i] = msg
			continue
		}

		strippedBlocks := stripMediaFromBlocks(blocks)

		// Create a shallow copy with stripped content
		newMsg := make(map[string]any, len(mm))
		for k, v := range mm {
			newMsg[k] = v
		}
		newMsg["content"] = strippedBlocks
		result[i] = newMsg
	}
	return result
}

// stripMediaFromBlocks replaces image and document blocks with text placeholders.
func stripMediaFromBlocks(blocks []any) []any {
	result := make([]any, 0, len(blocks))
	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			result = append(result, block)
			continue
		}

		blockType, _ := blk["type"].(string)
		switch blockType {
		case "image":
			result = append(result, map[string]any{
				"type": "text",
				"text": "[image]",
			})
		case "document":
			result = append(result, map[string]any{
				"type": "text",
				"text": "[document]",
			})
		default:
			result = append(result, block)
		}
	}
	return result
}

// summarize calls the summarization model with the old messages and returns
// the summary text. It uses the OpenAI chat/completions format, targeting
// the same upstream proxy that the main request would go to.
func (c *Compactor) summarize(ctx context.Context, body map[string]any, oldMessages []any) (string, error) {
	// Build OpenAI-format messages for the summarization request
	chatMessages := make([]map[string]any, 0, 2+len(oldMessages))

	// System prompt for the summarizer
	chatMessages = append(chatMessages, map[string]any{
		"role":    "system",
		"content": SummaryPrompt,
	})

	// Convert old Anthropic messages to a single user message for summarization
	// We marshal the conversation to JSON for a clean representation
	conversationJSON, err := json.MarshalIndent(oldMessages, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal old messages: %w", err)
	}

	chatMessages = append(chatMessages, map[string]any{
		"role":    "user",
		"content": UserSummaryPrompt + string(conversationJSON),
	})

	// Determine the model to use for summarization
	model := c.SummaryModel
	if model == "" {
		// Use the same model as the original request
		if m, ok := body["model"].(string); ok {
			model = m
		} else {
			model = "glm5.1"
		}
	}

	// Build the OpenAI chat completion request
	reqBody := map[string]any{
		"model":      model,
		"messages":   chatMessages,
		"max_tokens": 4096,
		"stream":     false,
	}

	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal summarization request: %w", err)
	}

	url := strings.TrimRight(c.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create summarization request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	// Use a timeout for summarization to avoid blocking the main request too long
	summarizeCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	req = req.WithContext(summarizeCtx)

	resp, err := c.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("summarization request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read summarization response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("summarization returned status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse OpenAI chat completion response
	var chatResp map[string]any
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", fmt.Errorf("failed to parse summarization response: %w", err)
	}

	choices, ok := chatResp["choices"].([]any)
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("summarization response has no choices")
	}

	firstChoice, ok := choices[0].(map[string]any)
	if !ok {
		return "", fmt.Errorf("invalid choice format in summarization response")
	}

	message, ok := firstChoice["message"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("no message in summarization response")
	}

	content, ok := message["content"].(string)
	if !ok || content == "" {
		return "", fmt.Errorf("empty content in summarization response")
	}

	return content, nil
}

// buildCompactedMessages constructs the compacted messages array by inserting
// the conversation summary while maintaining proper role alternation.
// Consecutive same-role messages (e.g., two "user" messages) violate both
// Anthropic and OpenAI conversation format requirements and cause 500 errors.
//
// Three cases based on the role of the first recent message:
//  1. "user"    → merge summary into the first user message (avoids user→user)
//  2. "assistant" → prepend summary as standalone user message (user→assistant)
//  3. tool_result → prepend summary as user + synthetic assistant with tool_calls
func buildCompactedMessages(summaryBlockText string, recentMessages []any) []any {
	if len(recentMessages) == 0 {
		return []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": summaryBlockText},
				},
			},
		}
	}

	firstMsg, _ := recentMessages[0].(map[string]any)
	firstRole, _ := firstMsg["role"].(string)

	// Detect tool_result in the first message. In Anthropic format, tool_result
	// blocks live inside a "user" role message, but after translateMessages in
	// litellm they become "tool" role messages that require a preceding assistant
	// with tool_calls. We must treat this as case 3 to avoid orphan tool messages.
	hasToolResult := firstRole == "tool" || containsToolResult(firstMsg)

	switch {
	case hasToolResult:
		// Case 3: First message has tool_result blocks (role is "user" but
		// content is only tool_result, or role is explicitly "tool"). After
		// translateMessages in litellm, these become "tool" role messages which
		// require a preceding assistant message with tool_calls. We also need a
		// user message before that assistant.
		// So: user(summary) → assistant(tool_calls) → recent...
		summaryMsg := map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": summaryBlockText},
			},
		}

		// Extract tool_use IDs from tool_result blocks to build the
		// synthetic assistant message with matching tool_calls.
		toolCalls := extractToolCallsFromToolResults(firstMsg)

		syntheticAssistant := map[string]any{
			"role":      "assistant",
			"content":   "I'll use the available tools.",
			"tool_calls": toolCalls,
		}

		result := make([]any, 0, 2+len(recentMessages))
		result = append(result, summaryMsg)
		result = append(result, syntheticAssistant)
		result = append(result, recentMessages...)
		return result

	case firstRole == "user":
		// Case 1: Merge summary into the first user message to avoid
		// two consecutive user messages.
		merged := mergeSummaryIntoMessage(firstMsg, summaryBlockText)
		result := make([]any, 0, len(recentMessages))
		result = append(result, merged)
		result = append(result, recentMessages[1:]...)
		return result

	default:
		// Case 2: Summary as user → assistant is valid alternation.
		summaryMsg := map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": summaryBlockText},
			},
		}
		result := make([]any, 0, 1+len(recentMessages))
		result = append(result, summaryMsg)
		result = append(result, recentMessages...)
		return result
	}
}

// mergeSummaryIntoMessage prepends a summary text block into a user message's
// content. It handles both string content and array content (Anthropic format).
// Returns a new message — does not modify the input.
func mergeSummaryIntoMessage(msg map[string]any, summaryText string) map[string]any {
	newMsg := make(map[string]any, len(msg))
	for k, v := range msg {
		newMsg[k] = v
	}

	summaryBlock := map[string]any{"type": "text", "text": summaryText}

	switch content := msg["content"].(type) {
	case string:
		// String content → convert to array with summary + original text
		newMsg["content"] = []any{
			summaryBlock,
			map[string]any{"type": "text", "text": content},
		}
	case []any:
		// Array content → prepend summary block
		newContent := make([]any, 0, 1+len(content))
		newContent = append(newContent, summaryBlock)
		newContent = append(newContent, content...)
		newMsg["content"] = newContent
	default:
		// Unknown content type → replace with summary only
		newMsg["content"] = []any{summaryBlock}
	}

	return newMsg
}

// containsToolResult checks whether a message's content contains any
// tool_result blocks. In Anthropic format, tool_result blocks are embedded
// in user-role messages but need special handling after translation.
func containsToolResult(msg map[string]any) bool {
	content, ok := msg["content"]
	if !ok {
		return false
	}

	blocks, ok := content.([]any)
	if !ok {
		return false
	}

	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			continue
		}
		if blk["type"] == "tool_result" {
			return true
		}
	}
	return false
}

// extractToolCallsFromToolResults scans a message's content for tool_result
// blocks and constructs a matching tool_calls array (OpenAI format) for a
// synthetic assistant message. This ensures that after translateMessages
// converts these tool_results into "tool" role messages, they have a valid
// preceding assistant message with tool_calls.
func extractToolCallsFromToolResults(msg map[string]any) []any {
	content, ok := msg["content"]
	if !ok {
		return nil
	}

	blocks, ok := content.([]any)
	if !ok {
		return nil
	}

	var toolCalls []any
	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			continue
		}
		if blk["type"] == "tool_result" {
			toolUseID, _ := blk["tool_use_id"].(string)
			// Anthropic tool_result doesn't carry the tool name;
			// use a generic placeholder.
			toolCalls = append(toolCalls, map[string]any{
				"id":   toolUseID,
				"type": "function",
				"function": map[string]any{
					"name":      "tool",
					"arguments": "{}",
				},
			})
		}
	}

	return toolCalls
}

// roleSequence returns a compact string representation of message roles
// for debugging, e.g., "user→assistant→user→assistant".
func roleSequence(messages []any) string {
	roles := make([]string, 0, len(messages))
	for _, msg := range messages {
		if mm, ok := msg.(map[string]any); ok {
			if role, ok := mm["role"].(string); ok {
				roles = append(roles, role)
			}
		}
	}
	return strings.Join(roles, "→")
}

// messageCount returns the number of messages in the request body.
func messageCount(body map[string]any) int {
	messagesRaw, ok := body["messages"]
	if !ok {
		return 0
	}
	msgArr, ok := messagesRaw.([]any)
	if !ok {
		return 0
	}
	return len(msgArr)
}
