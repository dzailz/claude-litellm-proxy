package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- defaultConfig tests ---

func TestDefaultConfig_ReturnsDeepSeekMode(t *testing.T) {
	cfg := defaultConfig()

	assert.Equal(t, BackendDeepSeek, cfg.BackendType)
	assert.Equal(t, "127.0.0.1:8082", cfg.ListenAddr)
	assert.Equal(t, "info", cfg.LogLevel)
	assert.Equal(t, "json", cfg.LogFormat)
	assert.Equal(t, "120s", cfg.ClientTimeout)
	assert.Equal(t, "https://api.deepseek.com/anthropic", cfg.DeepSeek.BaseURL)
	assert.Equal(t, "http://localhost:4000", cfg.LiteLLM.BaseURL)
	assert.Empty(t, cfg.DeepSeek.APIKey)
	assert.Empty(t, cfg.LiteLLM.APIKey)
	assert.Nil(t, cfg.LiteLLM.ModelMap)
}

func TestDefaultConfig_BackwardCompatibility_MatchesOldDeepSeekOnlyBehavior(t *testing.T) {
	// Zero-config defaults must mirror the original DeepSeek-only behavior
	// so existing deployments that don't set any config continue to work.
	cfg := defaultConfig()

	assert.Equal(t, BackendDeepSeek, cfg.BackendType,
		"default backend must be DeepSeek for backward compatibility")
	assert.Equal(t, "127.0.0.1:8082", cfg.ListenAddr,
		"default listen address must match original behavior")
	assert.Equal(t, "https://api.deepseek.com/anthropic", cfg.DeepSeek.BaseURL,
		"default DeepSeek base URL must match original behavior")
}

// --- LoadConfig tests: no config file ---

func TestLoadConfig_NoConfigFile_ReturnsDefaultsNoError(t *testing.T) {
	// Ensure CONFIG_FILE is not set in the environment.
	os.Unsetenv("CONFIG_FILE")

	cfg, err := LoadConfig("")

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendDeepSeek, cfg.BackendType)
	assert.Equal(t, "127.0.0.1:8082", cfg.ListenAddr)
	assert.Equal(t, "info", cfg.LogLevel)
	assert.Equal(t, "json", cfg.LogFormat)
}

// --- LoadConfig tests: CONFIG_FILE env var ---

func TestLoadConfig_CONFIG_FILE_EnvVar_PointsToValidYAML(t *testing.T) {
	yamlContent := `
backend:
  type: litellm
server:
  listen_addr: "0.0.0.0:9090"
logging:
  level: debug
  format: text
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	os.Setenv("CONFIG_FILE", path)
	defer os.Unsetenv("CONFIG_FILE")

	cfg, err := LoadConfig("")

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendLiteLLM, cfg.BackendType)
	assert.Equal(t, "0.0.0.0:9090", cfg.ListenAddr)
	assert.Equal(t, "debug", cfg.LogLevel)
	assert.Equal(t, "text", cfg.LogFormat)
}

func TestLoadConfig_CONFIG_FILE_EnvVar_PathArgTakesPrecedence(t *testing.T) {
	// When a path is passed explicitly, it should be used even if CONFIG_FILE is set.
	yamlPathContent := `
backend:
  type: deepseek
server:
  listen_addr: "127.0.0.1:3333"
`
	yamlEnvContent := `
backend:
  type: litellm
server:
  listen_addr: "0.0.0.0:9999"
`
	pathArg := writeTempYAML(t, yamlPathContent)
	defer os.Remove(pathArg)
	pathEnv := writeTempYAML(t, yamlEnvContent)
	defer os.Remove(pathEnv)

	os.Setenv("CONFIG_FILE", pathEnv)
	defer os.Unsetenv("CONFIG_FILE")

	cfg, err := LoadConfig(pathArg)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendDeepSeek, cfg.BackendType)
	assert.Equal(t, "127.0.0.1:3333", cfg.ListenAddr)
}

// --- LoadConfig tests: valid YAML ---

func TestLoadConfig_ValidYAML_DeepSeek_AllFields(t *testing.T) {
	// Prevent env var leakage from overriding YAML values.
	os.Unsetenv("DEEPSEEK_API_KEY")
	os.Unsetenv("DEEPSEEK_API_KEY_FILE")
	os.Unsetenv("LISTEN_ADDR")
	os.Unsetenv("LOG_LEVEL")
	os.Unsetenv("LOG_FORMAT")
	os.Unsetenv("CLIENT_TIMEOUT")
	os.Unsetenv("LITELLM_API_KEY")

	yamlContent := `
backend:
  type: deepseek
server:
  listen_addr: "0.0.0.0:8080"
  client_timeout: "60s"
logging:
  level: debug
  format: text
deepseek:
  api_key: "sk-test-key-123"
  base_url: "https://custom.deepseek.com/anthropic"
  api_key_file: "/tmp/deepseek-key"
litellm:
  base_url: "http://litellm:4000"
  api_key: "sk-litellm-key"
  model_map:
    claude-sonnet-4: "deepseek/deepseek-chat"
    claude-opus-4: "deepseek/deepseek-r1"
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendDeepSeek, cfg.BackendType)
	assert.Equal(t, "0.0.0.0:8080", cfg.ListenAddr)
	assert.Equal(t, "60s", cfg.ClientTimeout)
	assert.Equal(t, "debug", cfg.LogLevel)
	assert.Equal(t, "text", cfg.LogFormat)
	assert.Equal(t, "sk-test-key-123", cfg.DeepSeek.APIKey)
	assert.Equal(t, "https://custom.deepseek.com/anthropic", cfg.DeepSeek.BaseURL)
	assert.Equal(t, "/tmp/deepseek-key", cfg.DeepSeek.APIKeyFile)
	assert.Equal(t, "http://litellm:4000", cfg.LiteLLM.BaseURL)
	assert.Equal(t, "sk-litellm-key", cfg.LiteLLM.APIKey)
	assert.Equal(t, "deepseek/deepseek-chat", cfg.LiteLLM.ModelMap["claude-sonnet-4"])
	assert.Equal(t, "deepseek/deepseek-r1", cfg.LiteLLM.ModelMap["claude-opus-4"])
}

func TestLoadConfig_ValidYAML_LiteLLM_WithModelMap(t *testing.T) {
	yamlContent := `
backend:
  type: litellm
server:
  listen_addr: ":9090"
litellm:
  base_url: "https://litellm.example.com"
  api_key: "sk-lite"
  model_map:
    claude-sonnet-4: "openai/gpt-4o"
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendLiteLLM, cfg.BackendType)
	assert.Equal(t, ":9090", cfg.ListenAddr)
	assert.Equal(t, "https://litellm.example.com", cfg.LiteLLM.BaseURL)
	assert.Equal(t, "sk-lite", cfg.LiteLLM.APIKey)
	assert.Len(t, cfg.LiteLLM.ModelMap, 1)
	assert.Equal(t, "openai/gpt-4o", cfg.LiteLLM.ModelMap["claude-sonnet-4"])
}

func TestLoadConfig_PartialYAML_AppliesDefaultsForMissingFields(t *testing.T) {
	// Prevent env var leakage from overriding YAML values.
	os.Unsetenv("DEEPSEEK_API_KEY")
	os.Unsetenv("DEEPSEEK_API_KEY_FILE")

	// When YAML only specifies some fields, defaults should fill the rest.
	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key: "sk-override-only"
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendDeepSeek, cfg.BackendType)
	assert.Equal(t, "sk-override-only", cfg.DeepSeek.APIKey)
	// Unspecified fields should retain defaults.
	assert.Equal(t, "127.0.0.1:8082", cfg.ListenAddr)
	assert.Equal(t, "info", cfg.LogLevel)
	assert.Equal(t, "json", cfg.LogFormat)
	assert.Equal(t, "120s", cfg.ClientTimeout)
	assert.Equal(t, "https://api.deepseek.com/anthropic", cfg.DeepSeek.BaseURL)
}

// --- LoadConfig tests: invalid YAML ---

func TestLoadConfig_InvalidYAML_ReturnsError(t *testing.T) {
	path := writeTempYAML(t, `backend: type: deepseek: invalid::: yaml`)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	assert.Nil(t, cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load config from")
}

func TestLoadConfig_FileNotFound_ReturnsError(t *testing.T) {
	cfg, err := LoadConfig("/nonexistent/path/config.yaml")

	assert.Nil(t, cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to load config from")
}

// --- LoadConfig tests: validation ---

func TestLoadConfig_InvalidBackendType_ReturnsValidationError(t *testing.T) {
	yamlContent := `
backend:
  type: invalid
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	assert.Nil(t, cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid backend type")
	assert.Contains(t, err.Error(), "invalid")
	assert.Contains(t, err.Error(), "deepseek")
	assert.Contains(t, err.Error(), "litellm")
}

// --- LoadConfig tests: env var overrides ---

func TestLoadConfig_EnvVarOverrides_DEEPSEEK_API_KEY_And_LISTEN_ADDR(t *testing.T) {
	yamlContent := `
backend:
  type: deepseek
server:
  listen_addr: "127.0.0.1:9999"
deepseek:
  api_key: "sk-from-yaml"
  base_url: "https://yaml.deepseek.com/anthropic"
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	os.Setenv("DEEPSEEK_API_KEY", "sk-from-env")
	defer os.Unsetenv("DEEPSEEK_API_KEY")
	os.Setenv("LISTEN_ADDR", "0.0.0.0:5555")
	defer os.Unsetenv("LISTEN_ADDR")

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	// Env vars must take precedence over YAML values.
	assert.Equal(t, "sk-from-env", cfg.DeepSeek.APIKey)
	assert.Equal(t, "0.0.0.0:5555", cfg.ListenAddr)
	// YAML value that was not overridden should remain.
	assert.Equal(t, "https://yaml.deepseek.com/anthropic", cfg.DeepSeek.BaseURL)
}

func TestLoadConfig_EnvVarOverrides_AllServerAndAuthSettings(t *testing.T) {
	// Test that every env-var-overridable setting works.
	path := writeTempYAML(t, `backend: {type: deepseek}`)
	defer os.Remove(path)

	os.Setenv("LISTEN_ADDR", "10.0.0.1:1234")
	defer os.Unsetenv("LISTEN_ADDR")
	os.Setenv("LOG_LEVEL", "debug")
	defer os.Unsetenv("LOG_LEVEL")
	os.Setenv("LOG_FORMAT", "text")
	defer os.Unsetenv("LOG_FORMAT")
	os.Setenv("CLIENT_TIMEOUT", "30s")
	defer os.Unsetenv("CLIENT_TIMEOUT")
	os.Setenv("DEEPSEEK_API_KEY", "sk-env-ds")
	defer os.Unsetenv("DEEPSEEK_API_KEY")
	os.Setenv("DEEPSEEK_BASE_URL", "https://env.deepseek.com")
	defer os.Unsetenv("DEEPSEEK_BASE_URL")
	os.Setenv("DEEPSEEK_API_KEY_FILE", "/env/keyfile")
	defer os.Unsetenv("DEEPSEEK_API_KEY_FILE")
	os.Setenv("LITELLM_API_KEY", "sk-env-ll")
	defer os.Unsetenv("LITELLM_API_KEY")
	os.Setenv("LITELLM_BASE_URL", "https://env.litellm.com")
	defer os.Unsetenv("LITELLM_BASE_URL")

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "10.0.0.1:1234", cfg.ListenAddr)
	assert.Equal(t, "debug", cfg.LogLevel)
	assert.Equal(t, "text", cfg.LogFormat)
	assert.Equal(t, "30s", cfg.ClientTimeout)
	assert.Equal(t, "sk-env-ds", cfg.DeepSeek.APIKey)
	assert.Equal(t, "https://env.deepseek.com", cfg.DeepSeek.BaseURL)
	assert.Equal(t, "/env/keyfile", cfg.DeepSeek.APIKeyFile)
	assert.Equal(t, "sk-env-ll", cfg.LiteLLM.APIKey)
	assert.Equal(t, "https://env.litellm.com", cfg.LiteLLM.BaseURL)
}

func TestLoadConfig_EnvVarOverrides_EmptyEnvVarDoesNotOverride(t *testing.T) {
	// An empty env var should not override a YAML value.
	yamlContent := `
backend:
  type: deepseek
server:
  listen_addr: "yaml-addr:8080"
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	os.Setenv("LISTEN_ADDR", "")
	defer os.Unsetenv("LISTEN_ADDR")

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "yaml-addr:8080", cfg.ListenAddr,
		"empty env var should not override YAML value")
}

func TestLoadConfig_EnvVarOverrides_ApplyWithoutConfigFile(t *testing.T) {
	// Env var overrides should apply even when no config file exists.
	os.Unsetenv("CONFIG_FILE")
	os.Setenv("DEEPSEEK_API_KEY", "sk-no-file")
	defer os.Unsetenv("DEEPSEEK_API_KEY")
	os.Setenv("LISTEN_ADDR", ":9090")
	defer os.Unsetenv("LISTEN_ADDR")
	os.Setenv("LOG_LEVEL", "warn")
	defer os.Unsetenv("LOG_LEVEL")

	cfg, err := LoadConfig("")

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, BackendDeepSeek, cfg.BackendType)
	assert.Equal(t, "sk-no-file", cfg.DeepSeek.APIKey)
	assert.Equal(t, ":9090", cfg.ListenAddr)
	assert.Equal(t, "warn", cfg.LogLevel)
}

// --- LoadConfig tests: ${VAR_NAME} interpolation ---

func TestLoadConfig_Interpolation_EnvVarsInYAMLResolved(t *testing.T) {
	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key: ${TEST_DS_KEY}
  base_url: ${TEST_DS_URL}/anthropic
server:
  listen_addr: ${TEST_LISTEN}
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	os.Setenv("TEST_DS_KEY", "sk-interpolated")
	defer os.Unsetenv("TEST_DS_KEY")
	os.Setenv("TEST_DS_URL", "https://interp.deepseek.com")
	defer os.Unsetenv("TEST_DS_URL")
	os.Setenv("TEST_LISTEN", "1.2.3.4:7777")
	defer os.Unsetenv("TEST_LISTEN")

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "sk-interpolated", cfg.DeepSeek.APIKey)
	assert.Equal(t, "https://interp.deepseek.com/anthropic", cfg.DeepSeek.BaseURL)
	assert.Equal(t, "1.2.3.4:7777", cfg.ListenAddr)
}

func TestLoadConfig_Interpolation_UnsetVarBecomesEmpty(t *testing.T) {
	// Unset env vars referenced in YAML should become empty strings.
	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key: ${NONEXISTENT_VAR_12345}
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	// Ensure the var is definitely not set.
	os.Unsetenv("NONEXISTENT_VAR_12345")

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Empty(t, cfg.DeepSeek.APIKey,
		"unset env var in YAML should resolve to empty string")
}

// --- LoadConfig tests: api_key_file resolution ---

func TestLoadConfig_APIKeyFile_PopulatesAPIKey(t *testing.T) {
	keyFile := writeTempFile(t, "sk-from-keyfile\n")
	defer os.Remove(keyFile)

	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key_file: ` + keyFile + `
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "sk-from-keyfile", cfg.DeepSeek.APIKey,
		"api_key should be populated from api_key_file contents")
}

func TestLoadConfig_APIKeyFile_DirectKeyTakesPrecedence(t *testing.T) {
	// When both api_key and api_key_file are set, the direct key should NOT be overwritten.
	keyFile := writeTempFile(t, "sk-from-file\n")
	defer os.Remove(keyFile)

	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key: "sk-direct"
  api_key_file: ` + keyFile + `
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "sk-direct", cfg.DeepSeek.APIKey,
		"direct api_key must take precedence over api_key_file")
}

func TestLoadConfig_APIKeyFile_WhitespaceOnlyKey(t *testing.T) {
	// File with only whitespace should not populate api_key.
	keyFile := writeTempFile(t, "   \n\t  \n")
	defer os.Remove(keyFile)

	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key_file: ` + keyFile + `
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Empty(t, cfg.DeepSeek.APIKey,
		"whitespace-only key file should not set api_key")
}

func TestLoadConfig_APIKeyFile_FileNotFound_SilentlyIgnored(t *testing.T) {
	// A missing api_key_file is non-fatal — the key just stays empty.
	yamlContent := `
backend:
  type: deepseek
deepseek:
  api_key_file: /nonexistent/key/file
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Empty(t, cfg.DeepSeek.APIKey,
		"missing api_key_file should be silently ignored")
}

func TestLoadConfig_APIKeyFile_EnvVarAPIKeyFile(t *testing.T) {
	// DEEPSEEK_API_KEY_FILE env var should also trigger file resolution.
	keyFile := writeTempFile(t, "sk-env-keyfile\n")
	defer os.Remove(keyFile)

	yamlContent := `
backend:
  type: deepseek
`
	path := writeTempYAML(t, yamlContent)
	defer os.Remove(path)

	os.Setenv("DEEPSEEK_API_KEY_FILE", keyFile)
	defer os.Unsetenv("DEEPSEEK_API_KEY_FILE")

	cfg, err := LoadConfig(path)

	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "sk-env-keyfile", cfg.DeepSeek.APIKey,
		"api_key should be populated from file set via env var")
}

// --- interpolateEnvVars unit tests ---

func TestInterpolateEnvVars_NoVars_ReturnsUnchanged(t *testing.T) {
	os.Setenv("IRRELEVANT", "val")
	defer os.Unsetenv("IRRELEVANT")

	result := interpolateEnvVars("plain text without vars")

	assert.Equal(t, "plain text without vars", result)
}

func TestInterpolateEnvVars_SingleVar(t *testing.T) {
	os.Setenv("MY_VAR", "hello")
	defer os.Unsetenv("MY_VAR")

	result := interpolateEnvVars("prefix ${MY_VAR} suffix")

	assert.Equal(t, "prefix hello suffix", result)
}

func TestInterpolateEnvVars_MultipleVars(t *testing.T) {
	os.Setenv("A", "alpha")
	defer os.Unsetenv("A")
	os.Setenv("B", "beta")
	defer os.Unsetenv("B")

	result := interpolateEnvVars("${A}-${B}")

	assert.Equal(t, "alpha-beta", result)
}

func TestInterpolateEnvVars_UnsetVar(t *testing.T) {
	os.Unsetenv("MISSING")

	result := interpolateEnvVars("before ${MISSING} after")

	assert.Equal(t, "before  after", result,
		"unset var should be replaced with empty string")
}

func TestInterpolateEnvVars_AdjacentVars(t *testing.T) {
	os.Setenv("HOST", "localhost")
	defer os.Unsetenv("HOST")
	os.Setenv("PORT", "8080")
	defer os.Unsetenv("PORT")

	result := interpolateEnvVars("${HOST}:${PORT}")

	assert.Equal(t, "localhost:8080", result)
}

// --- resolveAPIKeyFile unit tests ---

func TestResolveAPIKeyFile_NoFileSet_DoesNothing(t *testing.T) {
	cfg := &Config{
		DeepSeek: DeepSeekBackendConfig{
			APIKey:     "",
			APIKeyFile: "",
		},
	}

	resolveAPIKeyFile(cfg)

	assert.Empty(t, cfg.DeepSeek.APIKey)
}

func TestResolveAPIKeyFile_KeyAlreadySet_DoesNotOverwrite(t *testing.T) {
	// Even if a file exists, the direct key must not be overwritten.
	keyFile := writeTempFile(t, "file-key\n")
	defer os.Remove(keyFile)

	cfg := &Config{
		DeepSeek: DeepSeekBackendConfig{
			APIKey:     "direct-key",
			APIKeyFile: keyFile,
		},
	}

	resolveAPIKeyFile(cfg)

	assert.Equal(t, "direct-key", cfg.DeepSeek.APIKey,
		"existing api_key must not be overwritten by file contents")
}

func TestResolveAPIKeyFile_FileMissing_SilentlyIgnored(t *testing.T) {
	cfg := &Config{
		DeepSeek: DeepSeekBackendConfig{
			APIKey:     "",
			APIKeyFile: "/does/not/exist",
		},
	}

	// Should not panic and should not set api_key.
	resolveAPIKeyFile(cfg)

	assert.Empty(t, cfg.DeepSeek.APIKey)
}

func TestResolveAPIKeyFile_TrimsWhitespace(t *testing.T) {
	keyFile := writeTempFile(t, "  \t trimmed-key  \n")
	defer os.Remove(keyFile)

	cfg := &Config{
		DeepSeek: DeepSeekBackendConfig{
			APIKey:     "",
			APIKeyFile: keyFile,
		},
	}

	resolveAPIKeyFile(cfg)

	assert.Equal(t, "trimmed-key", cfg.DeepSeek.APIKey)
}

// --- validate unit tests ---

func TestValidate_ValidBackendTypes(t *testing.T) {
	tests := []struct {
		name string
		bt   BackendType
	}{
		{"deepseek", BackendDeepSeek},
		{"litellm", BackendLiteLLM},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BackendType: tt.bt}
			err := cfg.validate()
			assert.NoError(t, err)
		})
	}
}

func TestValidate_InvalidBackendType(t *testing.T) {
	tests := []struct {
		name string
		bt   BackendType
	}{
		{"empty", ""},
		{"unknown", "unknown"},
		{"openai", "openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{BackendType: tt.bt}
			err := cfg.validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), "invalid backend type")
		})
	}
}

// --- helpers ---

// writeTempYAML creates a temporary YAML file with the given content
// and returns its path. The caller is responsible for cleanup.
func writeTempYAML(t *testing.T, content string) string {
	t.Helper()
	return writeTempFile(t, content)
}

// writeTempFile creates a temporary file with the given content
// and returns its path. The caller is responsible for cleanup.
func writeTempFile(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.yaml")
	err := os.WriteFile(path, []byte(content), 0644)
	require.NoError(t, err, "failed to write temp file")
	return path
}
