APP     := goose
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)
BUILD   := CGO_ENABLED=0 go build -trimpath -ldflags '$(LDFLAGS)'
BIN     := bin
DIST    := dist
CMD     := ./cmd/goose

PLATFORMS := \
	darwin/amd64 \
	darwin/arm64 \
	linux/amd64 \
	linux/arm64 \
	windows/amd64 \
	windows/arm64

.PHONY: all build clean test race cover vet vuln release fuzz conformance $(PLATFORMS)

all: build

build:
	$(BUILD) -o $(BIN)/$(APP) $(CMD)

test:
	go test ./...

race:
	go test -race ./...

cover:
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -1

vet:
	go vet ./...
	go vet -tags conformance ./internal/conformance/

vuln:
	@command -v govulncheck >/dev/null 2>&1 || go install golang.org/x/vuln/cmd/govulncheck@latest
	govulncheck ./...

clean:
	rm -rf $(BIN) $(DIST) coverage.out

release: $(PLATFORMS)
	@cd $(DIST) && shasum -a 256 $(APP)-* > checksums.txt
	@echo "release binaries in $(DIST)/"

$(PLATFORMS): OS   = $(word 1,$(subst /, ,$@))
$(PLATFORMS): ARCH = $(word 2,$(subst /, ,$@))
$(PLATFORMS): EXT  = $(if $(filter windows,$(OS)),.exe,)
$(PLATFORMS):
	GOOS=$(OS) GOARCH=$(ARCH) $(BUILD) -o $(DIST)/$(APP)-$(OS)-$(ARCH)$(EXT) $(CMD)

# Each fuzz target for a minute, seeded from the build cache. Every target runs
# even when one fails, and what it said is kept rather than thrown away.
fuzz:
	@mkdir -p $(DIST)
	@failed="" ; for t in \
	  "./wire FuzzDecode" \
	  "./wire FuzzAnthropicStreamStaysWellFormed" \
	  "./internal/wirecore FuzzScanner" \
	  "./internal/wirecore FuzzLines" \
	  "./internal/wirecore FuzzFold" \
	  "./internal/awsstream FuzzReader" \
	  "./llm FuzzStreamAccounting" \
	  "./cmd/goose FuzzLoadConfig" \
	  "./internal/roundtrip FuzzOpenAIRequestSurvivesTheProxy" \
	  "./internal/roundtrip FuzzAnthropicRequestSurvivesTheProxy" \
	  "./internal/roundtrip FuzzGoogleRequestSurvivesTheProxy" \
	  "./internal/roundtrip FuzzOllamaRequestSurvivesTheProxy" \
	  "./internal/roundtrip FuzzResponsesRequestSurvivesTheProxy" \
	  "./internal/roundtrip FuzzOpenAIAnswerSurvivesTheProxy" \
	  "./internal/roundtrip FuzzGoogleAnswerSurvivesTheProxy" \
	  "./internal/roundtrip FuzzOpenAIStreamSurvivesTheProxy" ; do \
	  set -- $$t ; \
	  printf '%-48s ' "$$2" ; \
	  if go test -run xxx -fuzz "$$2" -fuzztime 60s "$$1" >"$(DIST)/fuzz-$$2.log" 2>&1 ; then \
	    echo ok ; \
	  else \
	    echo "FAIL (see $(DIST)/fuzz-$$2.log)" ; failed="$$failed $$2" ; \
	  fi ; \
	done ; \
	if [ -n "$$failed" ] ; then echo "failed:$$failed" ; exit 1 ; fi

conformance:
	go test -tags conformance ./internal/conformance/ -v

