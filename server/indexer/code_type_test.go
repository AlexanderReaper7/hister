// SPDX-License-Identifier: AGPL-3.0-or-later

package indexer

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/asciimoo/hister/config"
	"github.com/asciimoo/hister/server/document"
	"github.com/asciimoo/hister/server/testutil"
	"github.com/asciimoo/hister/server/vectorstore"
)

func TestCodeDocumentKeepsTypeAndTimes(t *testing.T) {
	d := &document.Document{URL: "vscode://file/src/main.rs:10:1", Title: "repo/src/main.rs:10", Text: "fn main() {}", Type: document.Code, Added: 1000, Updated: 2000}
	if err := d.Process(nil, nil); err != nil {
		t.Fatal(err)
	}
	if d.Type != document.Code || d.Added != 1000 || d.Updated != 2000 {
		t.Fatalf("type, added, updated = %s, %d, %d, want code, 1000, 2000", d.Type, d.Added, d.Updated)
	}
}

func TestTypeFilterSeparatesCodeFromWeb(t *testing.T) {
	idx := newTestIndexer(t, testutil.Config(t))
	defer idx.Close()
	for _, d := range []*document.Document{
		{URL: "vscode://file/src/retry.rs:1:1", Title: "repo/src/retry.rs:1", Text: "retry upload", Type: document.Code},
		{URL: "https://example.com/retry", Title: "Retry", Text: "retry upload"},
	} {
		if err := d.Process(nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := idx.Add(d); err != nil {
			t.Fatal(err)
		}
	}
	for text, want := range map[string]string{
		"type:code retry": "vscode://file/src/retry.rs:1:1",
		"type:web retry":  "https://example.com/retry",
	} {
		result, err := idx.Search(&Query{Text: text})
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, d := range result.Documents {
			got = append(got, d.URL)
		}
		if !slices.Equal(got, []string{want}) {
			t.Errorf("%q: documents = %v, want [%s]", text, got, want)
		}
	}
}

func TestEmbeddedTextCarriesDatePerType(t *testing.T) {
	server, inputs := keywordEmbeddingServer(t)
	t.Cleanup(server.Close)
	cfg := testutil.Config(t)
	cfg.Server.Database = "hister-test.sqlite3"
	cfg.SemanticSearch.Enable = true
	cfg.SemanticSearch.EmbeddingEndpoint = server.URL
	cfg.SemanticSearch.EmbeddingModel = "test"
	cfg.SemanticSearch.Dimensions = 2
	cfg.SemanticSearch.MaxContextLength = 128
	cfg.SemanticSearch.MaxEmbeddingConcurrency = 1
	testutil.InitModelWithConfig(t, cfg)
	idx := newTestIndexer(t, cfg)
	defer idx.Close()

	const first, last int64 = 1758546180, 1759150980
	for _, d := range []*document.Document{
		{URL: "https://example.com/page", Title: "Page", Text: "web page text", Added: first, Updated: last},
		{URL: "vscode://file/src/lib.rs:1:1", Title: "repo/src/lib.rs:1", Text: "code piece text", Type: document.Code, Added: first, Updated: last},
	} {
		if err := d.Process(nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := idx.Add(d); err != nil {
			t.Fatal(err)
		}
	}
	waitForEmptyEmbeddingQueue(t)

	const layout = "Monday 2 January 2006, 15:04"
	wantWeb := "first visited: " + time.Unix(first, 0).Format(layout)
	wantCode := "modified: " + time.Unix(last, 0).Format(layout)
	var web, code []string
	for _, input := range inputs() {
		switch {
		case strings.Contains(input, "example.com/page") || strings.Contains(input, "web page text"):
			web = append(web, input)
		case strings.Contains(input, "src/lib.rs") || strings.Contains(input, "code piece text"):
			code = append(code, input)
		}
	}
	if len(web) == 0 || len(code) == 0 {
		t.Fatalf("no embedding inputs recorded: web %q, code %q", web, code)
	}
	for _, input := range web {
		if !strings.Contains(input, wantWeb) || strings.Contains(input, "modified:") {
			t.Errorf("web input %q: want %q and no modified date", input, wantWeb)
		}
	}
	for _, input := range code {
		if !strings.Contains(input, wantCode) || strings.Contains(input, "first visited:") {
			t.Errorf("code input %q: want %q and no first visit", input, wantCode)
		}
	}
}

func TestSearchReportsSemanticError(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
	}))
	defer server.Close()
	defer close(release)
	idx := newTestIndexer(t, testutil.Config(t))
	defer idx.Close()
	if err := idx.Add(&document.Document{URL: "https://example.com/slow", Title: "Slow", Text: "slow", Processed: true}); err != nil {
		t.Fatal(err)
	}
	idx.embedder = vectorstore.NewEmbedder(&config.SemanticSearch{
		EmbeddingEndpoint: server.URL, EmbeddingModel: "test", Dimensions: 3, QueryEmbeddingTimeout: 1,
	})
	idx.vectorStore = &metadataVectorStore{}
	result, err := idx.Search(&Query{Text: "slow", SemanticEnabled: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(result.SemanticError, "query embedding failed") || len(result.Documents) != 1 {
		t.Fatalf("semantic error %q with %d keyword documents, want a query embedding failure and 1", result.SemanticError, len(result.Documents))
	}
}
