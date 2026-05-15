// Package compaction provides server-side transparent conversation compaction
// for LLM proxy requests. When the estimated token count of a request's
// messages exceeds a configurable threshold, older messages are summarized
// into a compact form to fit within the upstream model's context window.
package compaction

import (
	"encoding/json"
)

// TokensPerChar is the rough ratio of tokens to characters for English text.
// Most tokenizers average ~4 characters per token for English.
const TokensPerChar = 4

// ImageTokenEstimate is the conservative token count estimate per image.
const ImageTokenEstimate = 1000

// DocumentTokenEstimate is the conservative token count estimate per document.
const DocumentTokenEstimate = 500

// EstimateTokens returns a rough token count for an Anthropic-format request
// body. It estimates by JSON-marshaling the relevant fields and dividing the
// byte count by TokensPerChar. Images and documents get fixed token estimates
// since their base64 payloads would massively inflate the JSON size.
//
// This is a fast heuristic — not precise — but sufficient for deciding
// whether compaction should be triggered.
func EstimateTokens(body map[string]any) int {
	total := 0

	// Count system prompt tokens
	if systemRaw, ok := body["system"]; ok {
		total += estimateValueTokens(systemRaw)
	}

	// Count message tokens
	if messagesRaw, ok := body["messages"]; ok {
		if msgArr, ok := messagesRaw.([]any); ok {
			for _, msg := range msgArr {
				total += estimateMessageTokens(msg)
			}
		}
	}

	// Count tool definition tokens
	if toolsRaw, ok := body["tools"]; ok {
		if toolsArr, ok := toolsRaw.([]any); ok {
			for _, tool := range toolsArr {
				total += estimateValueTokens(tool)
			}
		}
	}

	return total
}

// estimateMessageTokens returns the estimated token count for a single
// Anthropic-format message. It handles both string content and array
// content blocks, replacing images and documents with fixed estimates.
func estimateMessageTokens(msg any) int {
	mm, ok := msg.(map[string]any)
	if !ok {
		return estimateValueTokens(msg)
	}

	total := 0

	// Count role overhead (~1 token for "user"/"assistant"/"system")
	total += 1

	// Count content
	content, hasContent := mm["content"]
	if hasContent {
		total += estimateContentTokens(content)
	}

	// Count reasoning_content if present
	if rc, ok := mm["reasoning_content"]; ok && rc != nil {
		total += estimateValueTokens(rc)
	}

	return total
}

// estimateContentTokens estimates tokens for message content, which can be
// either a plain string or an array of content blocks.
func estimateContentTokens(content any) int {
	// String content: simple character-based estimate
	if str, ok := content.(string); ok {
		return charToTokens(len(str))
	}

	// Array content blocks
	blocks, ok := content.([]any)
	if !ok {
		return estimateValueTokens(content)
	}

	total := 0
	for _, block := range blocks {
		blk, ok := block.(map[string]any)
		if !ok {
			total += estimateValueTokens(block)
			continue
		}

		blockType, _ := blk["type"].(string)
		switch blockType {
		case "image":
			total += ImageTokenEstimate
		case "document":
			total += DocumentTokenEstimate
		case "text":
			if text, ok := blk["text"].(string); ok {
				total += charToTokens(len(text))
			}
		case "thinking":
			if thinking, ok := blk["thinking"].(string); ok {
				total += charToTokens(len(thinking))
			}
		case "tool_use":
			// Estimate from name + input JSON
			if name, ok := blk["name"].(string); ok {
				total += charToTokens(len(name))
			}
			if input := blk["input"]; input != nil {
				total += estimateValueTokens(input)
			}
		case "tool_result":
			if resultContent := blk["content"]; resultContent != nil {
				total += estimateContentTokens(resultContent)
			}
		default:
			// For unknown block types, estimate from the full block JSON
			total += estimateValueTokens(blk)
		}
	}

	return total
}

// estimateValueTokens marshals a value to JSON and estimates tokens from
// the resulting byte length.
func estimateValueTokens(v any) int {
	if v == nil {
		return 0
	}
	b, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	return charToTokens(len(b))
}

// charToTokens converts a character/byte count to an estimated token count.
func charToTokens(charCount int) int {
	if charCount <= 0 {
		return 0
	}
	return charCount/TokensPerChar + 1
}
