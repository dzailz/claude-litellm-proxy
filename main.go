package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"claude-go-to-deepseek-proxy/internal/logger"
	"claude-go-to-deepseek-proxy/internal/proxy"

	"github.com/go-chi/chi/v5"
)

func main() {
	listenAddr := getEnv("LISTEN_ADDR", "127.0.0.1:8082")

	apiKey, err := loadAPIKey()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}

	logLevel := getEnv("LOG_LEVEL", "info")
	logFormat := getEnv("LOG_FORMAT", "json")
	log := logger.Setup(logLevel, logFormat)

	timeoutStr := getEnv("CLIENT_TIMEOUT", "120s")
	timeout, err := time.ParseDuration(timeoutStr)
	if err != nil {
		log.Warn("invalid CLIENT_TIMEOUT, using default 120s", "value", timeoutStr)
		timeout = 120 * time.Second
	}

	handler := &proxy.ProxyHandler{
		UpstreamURL: "https://api.deepseek.com/anthropic",
		APIKey:      apiKey,
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
		"listen_addr", listenAddr,
		"log_level", logLevel,
		"log_format", logFormat,
		"connect_timeout", timeout.String(),
	)

	if err := http.ListenAndServe(listenAddr, r); err != nil {
		log.Error("server failed", "error", err)
		os.Exit(1)
	}
}

func loadAPIKey() (string, error) {
	if file := os.Getenv("DEEPSEEK_API_KEY_FILE"); file != "" {
		data, err := os.ReadFile(file)
		if err != nil {
			return "", fmt.Errorf("failed to read DEEPSEEK_API_KEY_FILE: %w", err)
		}
		key := strings.TrimSpace(string(data))
		if key == "" {
			return "", fmt.Errorf("DEEPSEEK_API_KEY_FILE is empty")
		}
		return key, nil
	}

	key := os.Getenv("DEEPSEEK_API_KEY")
	if key != "" {
		return key, nil
	}

	return "", fmt.Errorf("DEEPSEEK_API_KEY or DEEPSEEK_API_KEY_FILE must be set")
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}
