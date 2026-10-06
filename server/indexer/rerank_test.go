// SPDX-License-Identifier: AGPL-3.0-or-later

package indexer

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/asciimoo/hister/config"
	"github.com/asciimoo/hister/server/document"
	"github.com/asciimoo/hister/server/vectorstore"
)

func candidateIDs(cs []rerankCandidate) []string {
	ids := make([]string, len(cs))
	for j, c := range cs {
		ids[j] = c.docID
	}
	return ids
}

func candidates(ids ...string) []rerankCandidate {
	cs := make([]rerankCandidate, len(ids))
	for j, id := range ids {
		cs[j] = rerankCandidate{docID: id}
	}
	return cs
}

func TestSelectRerankCandidates(t *testing.T) {
	for _, c := range []struct {
		name              string
		keyword, semantic []rerankCandidate
		total, keywordN   int
		want              []string
	}{
		{"keyword first, then semantic", candidates("k1", "k2", "k3"), candidates("s1", "s2", "s3"), 4, 2, []string{"k1", "k2", "s1", "s2"}},
		{"a document in both counts once", candidates("k1", "b"), candidates("b", "s1", "s2"), 4, 2, []string{"k1", "b", "s1", "s2"}},
		{"keyword fills when semantic runs out", candidates("k1", "k2", "k3", "k4"), candidates("s1"), 4, 1, []string{"k1", "s1", "k2", "k3"}},
		{"semantic fills when keyword runs out", nil, candidates("s1", "s2", "s3"), 2, 1, []string{"s1", "s2"}},
		{"no keyword share", candidates("k1"), candidates("s1", "s2"), 2, 0, []string{"s1", "s2"}},
	} {
		if got := candidateIDs(selectRerankCandidates(c.keyword, c.semantic, c.total, c.keywordN)); !slices.Equal(got, c.want) {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

func TestRerankPassageCutsOnRuneBoundary(t *testing.T) {
	c := rerankCandidate{title: "T", text: "åäö"}
	// "T\n" is 2 bytes and every letter 2, so 5 bytes would end mid-letter.
	if got := c.passage(5); got != "T\nå" {
		t.Fatalf("passage = %q, want %q", got, "T\nå")
	}
	if got := c.passage(100); got != "T\nåäö" {
		t.Fatalf("passage = %q, want the whole text", got)
	}
}

func TestRerankerRejectsIncompleteScores(t *testing.T) {
	for name, body := range map[string]string{
		"missing":  `{"results":[{"index":0,"relevance_score":1}]}`,
		"repeated": `{"results":[{"index":0,"relevance_score":1},{"index":0,"relevance_score":2}]}`,
		"outside":  `{"results":[{"index":0,"relevance_score":1},{"index":2,"relevance_score":2}]}`,
	} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(body))
		}))
		r := vectorstore.NewReranker(&config.Rerank{Endpoint: server.URL, Timeout: 5})
		if _, err := r.Rerank(context.Background(), "q", []string{"a", "b"}); err == nil {
			t.Errorf("%s: no error", name)
		}
		server.Close()
	}
}

type rerankCall struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

// rerankServer scores a passage by the first word of scores it contains, so a
// test decides the order by content, and records each request.
func rerankServer(t *testing.T, status int, scores map[string]float64) (*httptest.Server, func() []rerankCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []rerankCall
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var call rerankCall
		if err := json.NewDecoder(r.Body).Decode(&call); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		calls = append(calls, call)
		mu.Unlock()
		if status != http.StatusOK {
			http.Error(w, "reranker down", status)
			return
		}
		type result struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
		}
		results := make([]result, len(call.Documents))
		for j, d := range call.Documents {
			results[j].Index = j
			for word, score := range scores {
				if strings.Contains(d, word) {
					results[j].RelevanceScore = score
				}
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"results": results})
	}))
	t.Cleanup(server.Close)
	return server, func() []rerankCall {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(calls)
	}
}

// newRerankTestIndexer adds two documents to newFilterTestIndexer's alpha
// and beta, which are semantic hits for "needle". gamma.example's text
// contains "needle", so it is a keyword hit, while its embedding is too far
// from the query's. alpha-delta.example is a semantic hit through its
// metadata chunk alone, the only chunk that holds the URL.
func newRerankTestIndexer(t *testing.T, rerankURL string) *Indexer {
	t.Helper()
	idx, _ := newFilterTestIndexer(t)
	for _, d := range []*document.Document{
		{URL: "https://gamma.example/page", Title: "Third", Text: "gamma needle haystack"},
		{URL: "https://alpha-delta.example/page", Title: "Fourth", Text: "delta text"},
	} {
		if err := d.Process(nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := idx.Add(d); err != nil {
			t.Fatal(err)
		}
	}
	waitForEmptyEmbeddingQueue(t)
	idx.semanticConfig.ResultLimit = 10
	idx.semanticConfig.Rerank = config.Rerank{Enable: true, Endpoint: rerankURL, Model: "test", Candidates: 30, KeywordCandidates: 10, MaxDocumentChars: 2000, Timeout: 5}
	return idx
}

func rerankedURLs(r *Results) []string {
	var urls []string
	for _, h := range r.Reranked {
		urls = append(urls, h.URL)
	}
	return urls
}

func TestSearchReranksKeywordAndSemanticHitsTogether(t *testing.T) {
	server, calls := rerankServer(t, http.StatusOK, map[string]float64{"First": 0.9, "gamma": 0.2})
	idx := newRerankTestIndexer(t, server.URL)
	result, err := idx.Search(&Query{Text: "needle", SemanticEnabled: true})
	if err != nil {
		t.Fatal(err)
	}
	// Candidates go keyword first (gamma), then by similarity, so this order
	// is the reranker's.
	if want := []string{"https://alpha.example/page", "https://gamma.example/page"}; len(result.Reranked) != 4 || !slices.Equal(rerankedURLs(result)[:2], want) {
		t.Fatalf("reranked = %v, want 4 starting with %v (rerank error %q)", rerankedURLs(result), want, result.RerankError)
	}
	if result.Reranked[0].RerankScore != 0.9 || result.Reranked[0].DocID == "" {
		t.Fatalf("first reranked hit = %+v, want score 0.9 and a doc_id", result.Reranked[0])
	}
	got := calls()
	if len(got) != 1 || got[0].Query != "needle" {
		t.Fatalf("rerank calls = %+v, want one for \"needle\"", got)
	}
	// A semantic hit is read by its best body chunk, or by its text when
	// only its metadata chunk matched, never by the metadata chunk.
	for _, want := range []string{"First\nalpha", "Fourth\ndelta text", "Third\ngamma needle haystack"} {
		if !slices.Contains(got[0].Documents, want) {
			t.Errorf("passages %q, want %q among them", got[0].Documents, want)
		}
	}
}

func TestSearchReportsRerankError(t *testing.T) {
	server, _ := rerankServer(t, http.StatusServiceUnavailable, nil)
	idx := newRerankTestIndexer(t, server.URL)
	result, err := idx.Search(&Query{Text: "needle", SemanticEnabled: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.RerankError, "503") || len(result.Reranked) != 0 {
		t.Fatalf("rerank error %q with %d reranked, want a 503 and none", result.RerankError, len(result.Reranked))
	}
	if len(result.Documents) == 0 || len(result.SemanticHits) == 0 {
		t.Fatalf("%d documents and %d semantic hits, want both kept", len(result.Documents), len(result.SemanticHits))
	}
}

func TestSearchDoesNotRerankAnotherSort(t *testing.T) {
	server, calls := rerankServer(t, http.StatusOK, nil)
	idx := newRerankTestIndexer(t, server.URL)
	for _, sort := range []string{"date", "-relevance"} {
		result, err := idx.Search(&Query{Text: "needle", SemanticEnabled: true, Sort: sort})
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Reranked) != 0 {
			t.Errorf("sort %q: reranked %v", sort, rerankedURLs(result))
		}
	}
	if n := len(calls()); n != 0 {
		t.Fatalf("%d rerank calls, want none", n)
	}
}
