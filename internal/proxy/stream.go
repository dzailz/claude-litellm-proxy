package proxy

import (
	"bufio"
	"io"
	"net/http"
)

func ProcessStream(body io.Reader, writer io.Writer) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	flusher, _ := writer.(http.Flusher)

	for scanner.Scan() {
		line := scanner.Text()

		if _, err := io.WriteString(writer, line+"\n"); err != nil {
			return err
		}

		if flusher != nil {
			flusher.Flush()
		}
	}

	return scanner.Err()
}
