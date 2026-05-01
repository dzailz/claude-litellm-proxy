FROM golang:1.23-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download && go mod verify

COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /proxy .

FROM alpine:3.20

RUN apk --no-cache add ca-certificates wget

COPY --from=builder /proxy /usr/local/bin/proxy

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8082/health || exit 1

ENTRYPOINT ["/usr/local/bin/proxy"]
