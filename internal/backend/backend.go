// Package backend defines the upstream API abstraction layer.
//
// The Backend interface enables clean separation between the proxy handler and
// the upstream service, supporting both simple passthrough (DeepSeek) and
// format-translating backends (LiteLLM) through a shared contract.
package backend

import (
	"context"
	"log/slog"
	"net/http"
)

// Backend abstracts an upstream API that accepts Anthropic-format requests.
// Implementations may forward requests directly (e.g., DeepSeek's
// native Anthropic-compatible API) or translate through an intermediary
// (e.g., LiteLLM, which converts Anthropic → OpenAI → Anthropic).
//
// Each method has a single, clearly defined responsibility so that
// concrete implementations remain small and testable.
type Backend interface {
	// PrepareRequest transforms the Anthropic-format request body into an
	// upstream HTTP request. The caller provides the parsed JSON body and
	// the incoming request headers; the implementation is responsible for
	// setting the target URL, method, body, and any backend-specific headers.
	//
	// The returned *http.Request must be ready to execute via http.Client.Do.
	// The caller owns the request and should set a context deadline/cancellation
	// before executing.
	PrepareRequest(ctx context.Context, body map[string]any, headers http.Header) (*http.Request, error)

	// NeedsStreamTransform reports whether streaming responses from this
	// backend require format transformation (e.g., OpenAI SSE → Anthropic SSE).
	// Returns false for backends that speak the Anthropic protocol natively
	// (passthrough), and true for backends that need translation.
	NeedsStreamTransform() bool

	// HandleResponse processes a non-streaming upstream response and writes
	// the result to the provided http.ResponseWriter. The implementation owns
	// the response body and must close it (via defer resp.Body.Close) after
	// reading.
	//
	// For passthrough backends this is a simple header+body copy. For
	// translation backends this transforms the response format before writing.
	HandleResponse(resp *http.Response, w http.ResponseWriter, logger *slog.Logger) error

	// HandleStream processes a streaming upstream response and writes the
	// Server-Sent Events (SSE) stream to the provided http.ResponseWriter.
	// The implementation owns the response body and must close it after the
	// stream ends or errors.
	//
	// For passthrough backends this copies SSE events directly. For
	// translation backends this transforms events (e.g., OpenAI SSE →
	// Anthropic SSE) in real time.
	HandleStream(resp *http.Response, w http.ResponseWriter, logger *slog.Logger) error
}
