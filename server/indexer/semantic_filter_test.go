// SPDX-License-Identifier: AGPL-3.0-or-later

package indexer

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/asciimoo/hister/server/document"
	"github.com/asciimoo/hister/server/model"
	"github.com/asciimoo/hister/server/testutil"
)

// keywordEmbeddingServer embeds by keyword, so a test can place documents at
// known similarities to the query "needle", which embeds as [1, 0] whatever
// its query prefix: "alpha" at 1.0, "beta" at 0.9, and anything else at 0.6.
// A document's metadata chunk carries its URL, so it scores like its body.
// It records every input.
func keywordEmbeddingServer(t *testing.T) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var inputs []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Input json.RawMessage `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var batch []string
		if len(request.Input) > 0 && request.Input[0] == '[' {
			if err := json.Unmarshal(request.Input, &batch); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		} else {
			var one string
			if err := json.Unmarshal(request.Input, &one); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			batch = []string{one}
		}
		mu.Lock()
		inputs = append(inputs, batch...)
		mu.Unlock()
		data := make([]map[string]any, len(batch))
		for j, text := range batch {
			vector := []float64{0.6, 0.8}
			switch {
			case strings.HasSuffix(text, "needle"), strings.Contains(text, "alpha"):
				vector = []float64{1, 0}
			case strings.Contains(text, "beta"):
				vector = []float64{0.9, 0.4359}
			}
			data[j] = map[string]any{"embedding": vector}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	return server, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(inputs)
	}
}

func waitForEmptyEmbeddingQueue(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		var jobs int64
		if err := model.DB.Model(&model.EmbeddingJob{}).Count(&jobs).Error; err != nil {
			t.Fatalf("count embedding jobs: %v", err)
		}
		if jobs == 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("embedding queue did not drain")
}

// newFilterTestIndexer indexes two web documents: alpha.example ranks first
// for "needle" and was updated in 2020, beta.example ranks second and was
// updated in 2026. result_limit is 1, so only a filter applied before ranking
// can return beta.example: unfiltered search returns alpha.example, and a
// filter applied to the single best hit returns nothing.
func newFilterTestIndexer(t *testing.T) (*Indexer, func() []string) {
	t.Helper()
	server, inputs := keywordEmbeddingServer(t)
	t.Cleanup(server.Close)
	cfg := testutil.Config(t)
	cfg.Server.Database = "hister-test.sqlite3"
	cfg.SemanticSearch.Enable = true
	cfg.SemanticSearch.EmbeddingEndpoint = server.URL
	cfg.SemanticSearch.EmbeddingModel = "test"
	cfg.SemanticSearch.Dimensions = 2
	cfg.SemanticSearch.MaxContextLength = 64
	cfg.SemanticSearch.ChunkOverlap = 4
	cfg.SemanticSearch.MaxEmbeddingConcurrency = 1
	cfg.SemanticSearch.ResultLimit = 1
	cfg.SemanticSearch.SimilarityThreshold = 0.85
	testutil.InitModelWithConfig(t, cfg)
	idx := newTestIndexer(t, cfg)
	t.Cleanup(func() { idx.Close() })

	for _, d := range []*document.Document{
		{URL: "https://alpha.example/page", Title: "First", Text: "alpha", Added: 1577836800, Updated: 1577836800},
		{URL: "https://beta.example/page", Title: "Second", Text: "beta", Added: 1758000000, Updated: 1758000000},
	} {
		if err := d.Process(nil, nil); err != nil {
			t.Fatalf("process %s: %v", d.URL, err)
		}
		if err := idx.Add(d); err != nil {
			t.Fatalf("add %s: %v", d.URL, err)
		}
	}
	waitForEmptyEmbeddingQueue(t)
	return idx, inputs
}

func semanticURLs(t *testing.T, idx *Indexer, q *Query) []string {
	t.Helper()
	q.SemanticEnabled = true
	result, err := idx.Search(q)
	if err != nil {
		t.Fatalf("search %q: %v", q.Text, err)
	}
	var urls []string
	for _, hit := range result.SemanticHits {
		urls = append(urls, hit.DocID)
	}
	return urls
}

func TestSemanticSearchUnfilteredReturnsBestHit(t *testing.T) {
	idx, _ := newFilterTestIndexer(t)
	got := semanticURLs(t, idx, &Query{Text: "needle"})
	if want := []string{"https://alpha.example/page"}; !slices.Equal(got, want) {
		t.Fatalf("semantic hits = %v, want %v", got, want)
	}
}

func TestSemanticSearchAppliesFiltersBeforeRanking(t *testing.T) {
	idx, _ := newFilterTestIndexer(t)
	for _, text := range []string{
		"domain:beta.example needle",
		"site:beta.example needle",
		"-domain:alpha.example needle",
		"updated:>=2025-01-01 needle",
		"(domain:beta.example|domain:other.example) needle",
	} {
		got := semanticURLs(t, idx, &Query{Text: text})
		if want := []string{"https://beta.example/page"}; !slices.Equal(got, want) {
			t.Errorf("%q: semantic hits = %v, want %v", text, got, want)
		}
	}
}

func TestSemanticSearchAppliesAPIDateRange(t *testing.T) {
	idx, _ := newFilterTestIndexer(t)
	got := semanticURLs(t, idx, &Query{Text: "needle", DateFrom: 1735689600})
	if want := []string{"https://beta.example/page"}; !slices.Equal(got, want) {
		t.Fatalf("semantic hits = %v, want %v", got, want)
	}
}

func TestSemanticSearchFilterMatchingNothingReturnsNoHits(t *testing.T) {
	idx, _ := newFilterTestIndexer(t)
	if got := semanticURLs(t, idx, &Query{Text: "domain:missing.example needle"}); len(got) != 0 {
		t.Fatalf("semantic hits = %v, want none", got)
	}
}

func TestSemanticSearchDoesNotEmbedFilters(t *testing.T) {
	idx, inputs := newFilterTestIndexer(t)
	before := len(inputs())
	semanticURLs(t, idx, &Query{Text: "domain:beta.example needle"})
	queries := inputs()[before:]
	if len(queries) != 1 || !strings.HasSuffix(queries[0], "needle") || strings.Contains(queries[0], "domain:") {
		t.Fatalf("embedded query inputs = %q, want one input ending in \"needle\" without the filter", queries)
	}
}

func TestAddKeepsEarliestAddedAndSubmittedUpdated(t *testing.T) {
	idx := newTestIndexer(t, testutil.Config(t))
	defer idx.Close()
	const url = "https://example.com/timestamps"
	for _, times := range [][2]int64{{2000, 2500}, {1000, 3000}, {1500, 3500}} {
		d := &document.Document{URL: url, Title: "Timestamps", Text: "timestamps", Added: times[0], Updated: times[1]}
		if err := d.Process(nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := idx.Add(d); err != nil {
			t.Fatal(err)
		}
	}
	got := idx.GetByURLAndUser(url, 0)
	if got == nil {
		t.Fatal("document missing")
	}
	if got.Added != 1000 || got.Updated != 3500 {
		t.Fatalf("added, updated = %d, %d, want 1000, 3500", got.Added, got.Updated)
	}
}
