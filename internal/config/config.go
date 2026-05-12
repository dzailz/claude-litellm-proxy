// Package config provides YAML-based configuration loading with env var
// overrides for the proxy application. It supports two backend modes:
// "deepseek" (native Anthropic-compatible API) and "litellm" (OpenAI bridge).
//
// Configuration can be supplied via a YAML file (path from CONFIG_FILE env var
// or passed directly to LoadConfig) and overridden by environment variables.
// When no config file is available, the package returns zero-config defaults
// that mirror the original DeepSeek-only behavior.
package config

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

// BackendType represents the upstream proxy backend mode.
type BackendType string

const (
	// BackendDeepSeek uses the DeepSeek native Anthropic-compatible API.
	BackendDeepSeek BackendType = "deepseek"
	// BackendLiteLLM routes through a LiteLLM proxy for format translation.
	BackendLiteLLM BackendType = "litellm"
)

// DeepSeekBackendConfig holds DeepSeek-specific configuration.
type DeepSeekBackendConfig struct {
	// APIKey is the DeepSeek API authentication key.
	APIKey string `yaml:"api_key"`
	// APIKeyFile is a path to a file containing the API key.
	// If set and non-empty, the file contents are read into APIKey.
	APIKeyFile string `yaml:"api_key_file"`
	// BaseURL is the DeepSeek API endpoint (without trailing path).
	BaseURL string `yaml:"base_url"`
}

// LiteLLMBackendConfig holds LiteLLM-specific configuration.
type LiteLLMBackendConfig struct {
	// BaseURL is the LiteLLM proxy endpoint.
	BaseURL string `yaml:"base_url"`
	// APIKey is the LiteLLM authentication key.
	APIKey string `yaml:"api_key"`
	// ModelMap translates Claude model names to LiteLLM model identifiers
	// (e.g., "deepseek-v4-pro" → "deepseek/deepseek-chat").
	ModelMap map[string]string `yaml:"model_map"`
}

// Config holds the full application configuration with flattened fields
// for common settings and nested structs for backend-specific options.
type Config struct {
	// BackendType is "deepseek" or "litellm".
	BackendType BackendType
	// ListenAddr is the address the HTTP server binds to.
	ListenAddr string
	// LogLevel controls the slog logging level ("debug", "info", "warn", "error").
	LogLevel string
	// LogFormat controls the log output format ("json" or "text").
	LogFormat string
	// ClientTimeout is the HTTP client timeout duration string (e.g., "120s").
	ClientTimeout string

	// DeepSeek holds DeepSeek-specific configuration.
	DeepSeek DeepSeekBackendConfig
	// LiteLLM holds LiteLLM-specific configuration.
	LiteLLM LiteLLMBackendConfig
}

// fileConfig mirrors the YAML config file structure for deserialization.
type fileConfig struct {
	Backend struct {
		Type string `yaml:"type"`
	} `yaml:"backend"`
	Server struct {
		ListenAddr    string `yaml:"listen_addr"`
		ClientTimeout string `yaml:"client_timeout"`
	} `yaml:"server"`
	Logging struct {
		Level  string `yaml:"level"`
		Format string `yaml:"format"`
	} `yaml:"logging"`
	DeepSeek DeepSeekBackendConfig `yaml:"deepseek"`
	LiteLLM  LiteLLMBackendConfig  `yaml:"litellm"`
}

// envVarRe matches ${VAR_NAME} patterns in config values.
var envVarRe = regexp.MustCompile(`\$\{([^}]+)\}`)

// LoadConfig loads configuration from the YAML file at the given path.
// If path is empty, the CONFIG_FILE environment variable is used.
// When no config file is available, zero-config defaults are returned.
//
// Env var references (${VAR_NAME}) in YAML values are interpolated from
// the environment. After parsing, direct environment variables override
// any YAML values, giving runtime configuration the highest priority.
//
// Returns an error only when a config file is explicitly specified (via
// path or CONFIG_FILE) and cannot be read or parsed.
func LoadConfig(path string) (*Config, error) {
	if path == "" {
		path = os.Getenv("CONFIG_FILE")
	}

	cfg := defaultConfig()

	if path == "" {
		return applyEnvOverrides(cfg), nil
	}

	fc, err := parseConfigFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load config from %s: %w", path, err)
	}

	cfg = fileConfigToConfig(fc)
	cfg = applyEnvOverrides(cfg)

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

// defaultConfig returns a zero-config configuration that mirrors the
// original DeepSeek-only defaults for backward compatibility.
func defaultConfig() *Config {
	return &Config{
		BackendType:   BackendDeepSeek,
		ListenAddr:    "127.0.0.1:8082",
		LogLevel:      "info",
		LogFormat:     "json",
		ClientTimeout: "120s",
		DeepSeek: DeepSeekBackendConfig{
			BaseURL: "https://api.deepseek.com/anthropic",
		},
		LiteLLM: LiteLLMBackendConfig{
			BaseURL: "http://localhost:4000",
		},
	}
}

// parseConfigFile reads a YAML config file and performs env var interpolation
// on ${VAR_NAME} references before unmarshalling.
func parseConfigFile(path string) (*fileConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	interpolated := interpolateEnvVars(string(data))

	var fc fileConfig
	if err := yaml.Unmarshal([]byte(interpolated), &fc); err != nil {
		return nil, fmt.Errorf("failed to parse config YAML: %w", err)
	}

	return &fc, nil
}

// interpolateEnvVars replaces ${VAR_NAME} patterns with the value
// of the corresponding environment variable. Unset variables are
// replaced with an empty string (consistent with shell behavior).
func interpolateEnvVars(input string) string {
	return envVarRe.ReplaceAllStringFunc(input, func(match string) string {
		varName := match[2 : len(match)-1] // strip "${" and "}"
		return os.Getenv(varName)
	})
}

// fileConfigToConfig converts the nested YAML structure into the flat
// Config struct, applying only non-empty values on top of defaults.
func fileConfigToConfig(fc *fileConfig) *Config {
	cfg := defaultConfig()

	if fc.Backend.Type != "" {
		cfg.BackendType = BackendType(fc.Backend.Type)
	}
	if fc.Server.ListenAddr != "" {
		cfg.ListenAddr = fc.Server.ListenAddr
	}
	if fc.Server.ClientTimeout != "" {
		cfg.ClientTimeout = fc.Server.ClientTimeout
	}
	if fc.Logging.Level != "" {
		cfg.LogLevel = fc.Logging.Level
	}
	if fc.Logging.Format != "" {
		cfg.LogFormat = fc.Logging.Format
	}
	if fc.DeepSeek.APIKey != "" {
		cfg.DeepSeek.APIKey = fc.DeepSeek.APIKey
	}
	if fc.DeepSeek.APIKeyFile != "" {
		cfg.DeepSeek.APIKeyFile = fc.DeepSeek.APIKeyFile
	}
	if fc.DeepSeek.BaseURL != "" {
		cfg.DeepSeek.BaseURL = fc.DeepSeek.BaseURL
	}
	if fc.LiteLLM.BaseURL != "" {
		cfg.LiteLLM.BaseURL = fc.LiteLLM.BaseURL
	}
	if fc.LiteLLM.APIKey != "" {
		cfg.LiteLLM.APIKey = fc.LiteLLM.APIKey
	}
	if fc.LiteLLM.ModelMap != nil && len(fc.LiteLLM.ModelMap) > 0 {
		cfg.LiteLLM.ModelMap = fc.LiteLLM.ModelMap
	}

	return cfg
}

// applyEnvOverrides applies direct environment variable overrides on top
// of the parsed config, giving env vars the highest priority. It also
// resolves the api_key_file → api_key chain when a key file is specified.
func applyEnvOverrides(cfg *Config) *Config {
	// Server settings
	if v := os.Getenv("LISTEN_ADDR"); v != "" {
		cfg.ListenAddr = v
	}
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	if v := os.Getenv("LOG_FORMAT"); v != "" {
		cfg.LogFormat = v
	}
	if v := os.Getenv("CLIENT_TIMEOUT"); v != "" {
		cfg.ClientTimeout = v
	}

	// DeepSeek auth — env vars take precedence
	if v := os.Getenv("DEEPSEEK_API_KEY"); v != "" {
		cfg.DeepSeek.APIKey = v
	}
	if v := os.Getenv("DEEPSEEK_API_KEY_FILE"); v != "" {
		cfg.DeepSeek.APIKeyFile = v
	}
	if v := os.Getenv("DEEPSEEK_BASE_URL"); v != "" {
		cfg.DeepSeek.BaseURL = v
	}

	// LiteLLM auth
	if v := os.Getenv("LITELLM_API_KEY"); v != "" {
		cfg.LiteLLM.APIKey = v
	}
	if v := os.Getenv("LITELLM_BASE_URL"); v != "" {
		cfg.LiteLLM.BaseURL = v
	}

	// Resolve api_key_file to api_key when a file path is set but
	// no direct key was provided (mirrors the loadAPIKey pattern
	// from main.go for backward compatibility).
	resolveAPIKeyFile(cfg)

	return cfg
}

// resolveAPIKeyFile reads the api_key_file path and populates api_key
// when a file is specified but no direct key was provided.
func resolveAPIKeyFile(cfg *Config) {
	if cfg.DeepSeek.APIKeyFile == "" || cfg.DeepSeek.APIKey != "" {
		return
	}
	data, err := os.ReadFile(cfg.DeepSeek.APIKeyFile)
	if err != nil {
		return // file read failure is non-fatal; caller validates later
	}
	key := strings.TrimSpace(string(data))
	if key != "" {
		cfg.DeepSeek.APIKey = key
	}
}

// validate checks that the config values meet the required constraints.
func (c *Config) validate() error {
	switch c.BackendType {
	case BackendDeepSeek, BackendLiteLLM:
		// valid
	default:
		return fmt.Errorf("invalid backend type: %q (must be %q or %q)",
			c.BackendType, BackendDeepSeek, BackendLiteLLM)
	}
	return nil
}
