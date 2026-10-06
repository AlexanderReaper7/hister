// SPDX-License-Identifier: AGPL-3.0-or-later

package vectorstore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/asciimoo/hister/config"
)

// Reranker calls a /v1/rerank endpoint, the API of llama-server, Jina and
// Cohere: a query and a list of documents in, a relevance score per document
// out.
type Reranker struct {
	endpoint string
	model    string
	apiKey   string
	headers  map[string]string
	client   *http.Client
}

func NewReranker(cfg *config.Rerank) *Reranker {
	return &Reranker{
		endpoint: cfg.Endpoint,
		model:    cfg.Model,
		apiKey:   cfg.APIKey,
		headers:  cfg.Headers,
		client:   &http.Client{Timeout: time.Duration(cfg.Timeout) * time.Second},
	}
}

type rerankRequest struct {
	Model     string   `json:"model,omitempty"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rerank returns one score per document, in the order the documents were
// given. A response that leaves a document out, or names one twice, is an
// error rather than a partial order.
func (r *Reranker) Rerank(ctx context.Context, query string, documents []string) (_ []float64, err error) {
	body, err := json.Marshal(rerankRequest{Model: r.model, Query: query, Documents: documents})
	if err != nil {
		return nil, fmt.Errorf("marshal rerank request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", r.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if r.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
	}
	for k, v := range r.headers {
		req.Header.Set(k, v)
	}

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank request failed: %w", err)
	}
	defer func() {
		if cerr := resp.Body.Close(); err == nil {
			err = cerr
		}
	}()
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("rerank endpoint returned %d: %s", resp.StatusCode, respBody)
	}

	var result rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode rerank response: %w", err)
	}
	if len(result.Results) != len(documents) {
		return nil, fmt.Errorf("rerank endpoint scored %d of %d documents", len(result.Results), len(documents))
	}
	scores := make([]float64, len(documents))
	seen := make([]bool, len(documents))
	for _, res := range result.Results {
		if res.Index < 0 || res.Index >= len(documents) || seen[res.Index] {
			return nil, fmt.Errorf("rerank endpoint returned an invalid or repeated index %d", res.Index)
		}
		seen[res.Index] = true
		scores[res.Index] = res.RelevanceScore
	}
	return scores, nil
}
