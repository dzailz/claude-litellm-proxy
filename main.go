package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"claude-go-to-deepseek-proxy/internal/backend"
	"claude-go-to-deepseek-proxy/internal/compaction"
	"claude-go-to-deepseek-proxy/internal/config"
	"claude-go-to-deepseek-proxy/internal/logger"
	"claude-go-to-deepseek-proxy/internal/proxy"

	"github.com/go-chi/chi/v5"
)

func main() {
	// Load configuration via config.LoadConfig, which reads from CONFIG_FILE
	// env var, YAML file, and env var overrides in layered priority. When
	// no config file is available, zero-config defaults are returned
	// (DeepSeek mode with standard settings for backward compatibility).
	cfg, err := config.LoadConfig("")
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}

	// Initialize structured logger from config settings.
	log := logger.Setup(cfg.LogLevel, cfg.LogFormat)

	// Parse client timeout with fallback to a safe default on invalid values.
	timeout, err := time.ParseDuration(cfg.ClientTimeout)
	if err != nil {
		log.Warn("invalid client timeout, using default 120s", "value", cfg.ClientTimeout)
		timeout = 120 * time.Second
	}

	// Create the appropriate backend based on config.BackendType.
	// DeepSeek: native Anthropic-compatible API with API key auth.
	// LiteLLM: OpenAI bridge that translates formats.
	// We also capture the backend connection details for compaction setup.
	var backendImpl backend.Backend
	var compactionBaseURL, compactionAPIKey, compactionAPIFormat string

	switch cfg.BackendType {
	case config.BackendDeepSeek:
		apiKey := cfg.DeepSeek.APIKey
		if apiKey == "" {
			// Fall back to legacy env vars for backward compatibility.
			if file := os.Getenv("DEEPSEEK_API_KEY_FILE"); file != "" {
				data, err := os.ReadFile(file)
				if err == nil {
					apiKey = strings.TrimSpace(string(data))
				}
			}
			if apiKey == "" {
				apiKey = os.Getenv("DEEPSEEK_API_KEY")
			}
		}
		if apiKey == "" {
			fmt.Fprintf(os.Stderr, "fatal: DeepSeek API key not configured\n")
			os.Exit(1)
		}
		backendImpl = &backend.DeepSeekBackend{
			BaseURL: cfg.DeepSeek.BaseURL,
			APIKey:  apiKey,
		}
		compactionBaseURL = cfg.DeepSeek.BaseURL
		compactionAPIKey = apiKey
		compactionAPIFormat = "anthropic"

	case config.BackendLiteLLM:
		apiKey := cfg.LiteLLM.APIKey
		if apiKey == "" {
			apiKey = os.Getenv("LITELLM_API_KEY")
		}
		// LiteLLM can operate without an API key in no-auth mode,
		// so an empty key is not a fatal condition.
		backendImpl = &backend.LiteLLMBackend{
			BaseURL:  cfg.LiteLLM.BaseURL,
			APIKey:   apiKey,
			ModelMap: cfg.LiteLLM.ModelMap,
		}
		compactionBaseURL = cfg.LiteLLM.BaseURL
		compactionAPIKey = apiKey
		compactionAPIFormat = "openai"

	default:
		fmt.Fprintf(os.Stderr, "fatal: unsupported backend type: %s\n", cfg.BackendType)
		os.Exit(1)
	}

	// Compose the ProxyHandler with the selected backend and HTTP client.
	// The HTTP client is owned by the handler; backends only prepare
	// requests and handle responses without owning the transport.
	handler := &proxy.ProxyHandler{
		Backend: backendImpl,
		HTTPClient: &http.Client{
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: timeout,
				}).DialContext,
				TLSHandshakeTimeout:   timeout,
				ResponseHeaderTimeout: timeout,
				IdleConnTimeout:       90 * time.Second,
			},
		},
		Logger: log,
	}

	// Initialize conversation compactor if enabled. The compactor uses the
	// same upstream proxy as the main request, but with a dedicated HTTP
	// client and potentially a different (cheaper/faster) model for
	// summarization. When disabled, Compactor remains nil and compaction
	// is skipped entirely.
	if cfg.Compaction.Enabled {
		handler.Compactor = &compaction.Compactor{
			Enabled:      true,
			MaxTokens:    cfg.Compaction.MaxTokens,
			SummaryModel: cfg.Compaction.SummaryModel,
			MinMessages:  cfg.Compaction.MinMessages,
			BaseURL:      compactionBaseURL,
			APIKey:       compactionAPIKey,
			APIFormat:    compactionAPIFormat,
			Client: &http.Client{
				Timeout: 90 * time.Second,
			},
			Logger: log,
		}
		log.Info("compaction enabled",
			"max_tokens", cfg.Compaction.MaxTokens,
			"summary_model", cfg.Compaction.SummaryModel,
			"min_messages", cfg.Compaction.MinMessages,
			"api_format", compactionAPIFormat,
		)
	}

	// Router setup — unchanged from previous version.
	r := chi.NewRouter()

	r.Post("/v1/messages", handler.ServeHTTP)
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"error":"not found"}`))
	})

	log.Info("starting proxy server",
		"listen_addr", cfg.ListenAddr,
		"backend_type", cfg.BackendType,
		"log_level", cfg.LogLevel,
		"log_format", cfg.LogFormat,
		"connect_timeout", timeout.String(),
		"compaction_enabled", cfg.Compaction.Enabled,
	)

	if err := http.ListenAndServe(cfg.ListenAddr, r); err != nil {
		log.Error("server failed", "error", err)
		os.Exit(1)
	}
}
