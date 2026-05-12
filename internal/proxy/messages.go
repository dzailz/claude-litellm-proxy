package proxy

import "claude-go-to-deepseek-proxy/internal/backend"

// RemoveRedactedThinking delegates to backend.RemoveRedactedThinking.
// Thin wrapper maintained for backward compatibility during the
// transition to the backend abstraction layer.
func RemoveRedactedThinking(messages []map[string]any) ([]map[string]any, int) {
	return backend.RemoveRedactedThinking(messages)
}

// ShouldKeepReasoningContent delegates to backend.ShouldKeepReasoningContent.
func ShouldKeepReasoningContent(messages []map[string]any) bool {
	return backend.ShouldKeepReasoningContent(messages)
}

// StripReasoningContent delegates to backend.StripReasoningContent.
func StripReasoningContent(messages []map[string]any) int {
	return backend.StripReasoningContent(messages)
}

// CountThinkingWithoutSignature delegates to backend.CountThinkingWithoutSignature.
func CountThinkingWithoutSignature(messages []map[string]any) int {
	return backend.CountThinkingWithoutSignature(messages)
}

// isUnsupportedContentType delegates to backend.IsUnsupportedContentType.
func isUnsupportedContentType(t any) bool {
	return backend.IsUnsupportedContentType(t)
}
