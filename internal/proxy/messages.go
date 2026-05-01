package proxy

var unsupportedContentTypes = map[string]bool{
	"redacted_thinking":           true,
	"server_tool_use":             true,
	"web_search_tool_result":      true,
	"code_execution_tool_result":  true,
	"mcp_tool_use":                true,
	"mcp_tool_result":             true,
	"container_upload":            true,
	"image":                       true,
	"document":                    true,
	"search_result":               true,
}

func isUnsupportedContentType(t any) bool {
	s, ok := t.(string)
	if !ok {
		return false
	}
	return unsupportedContentTypes[s]
}

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
			if isUnsupportedContentType(blk["type"]) {
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

func copyMessage(m map[string]any) map[string]any {
	cp := make(map[string]any, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

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
