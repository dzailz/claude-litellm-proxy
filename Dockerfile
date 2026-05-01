FROM golang:1.23-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o proxy .

FROM alpine:3.20
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/proxy /usr/local/bin/
EXPOSE 8082
ENTRYPOINT ["/usr/local/bin/proxy"]
