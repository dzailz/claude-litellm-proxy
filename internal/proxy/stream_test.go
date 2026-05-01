package proxy

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProcessStream_EmptyStream(t *testing.T) {
	reader := strings.NewReader("")
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.Empty(t, buf.String())
}

func TestProcessStream_SingleEvent(t *testing.T) {
	input := "data: {\"type\":\"content_block_delta\"}\n\ndata: [DONE]\n\n"
	reader := strings.NewReader(input)
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.Equal(t, input, buf.String())
}

func TestProcessStream_MultipleEvents(t *testing.T) {
	input := "data: {\"delta\":1}\n\ndata: {\"delta\":2}\n\ndata: {\"delta\":3}\n\ndata: [DONE]\n\n"
	reader := strings.NewReader(input)
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.Equal(t, input, buf.String())
}

func TestProcessStream_DoneMarker(t *testing.T) {
	input := "data: [DONE]\n\n"
	reader := strings.NewReader(input)
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.Equal(t, input, buf.String())
}

func TestProcessStream_PartialLine(t *testing.T) {
	input := "data: {\"type\":\"incomplete"
	reader := strings.NewReader(input)
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.Equal(t, input+"\n", buf.String())
}

func TestProcessStream_ThinkingBlock(t *testing.T) {
	input := "data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"thinking\"}}\n\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"thinking\":\"Let me think...\"}}\n\ndata: [DONE]\n\n"
	reader := strings.NewReader(input)
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.Equal(t, input, buf.String())
}

func TestProcessStream_WriterError(t *testing.T) {
	reader := strings.NewReader("data: hello\n\n")
	errWriter := &errorWriter{err: errors.New("write failed")}
	err := ProcessStream(reader, errWriter)
	assert.Error(t, err)
}

func TestProcessStream_LargeStream(t *testing.T) {
	var input strings.Builder
	for i := 0; i < 100; i++ {
		input.WriteString("data: {\"delta\":" + strings.Repeat("x", 1000) + "}\n\n")
	}
	reader := strings.NewReader(input.String())
	var buf bytes.Buffer
	err := ProcessStream(reader, &buf)
	require.NoError(t, err)
	assert.NotEmpty(t, buf.String())
}

type errorWriter struct {
	err error
}

func (w *errorWriter) Write(p []byte) (int, error) {
	return 0, w.err
}

var _ io.Writer = (*errorWriter)(nil)
