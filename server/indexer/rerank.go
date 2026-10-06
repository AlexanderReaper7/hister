// SPDX-License-Identifier: AGPL-3.0-or-later

package indexer

import (
	"context"
	"slices"
	"time"
	"unicode/utf8"

	"github.com/blevesearch/bleve/v2/search"
	"github.com/rs/zerolog/log"

	"github.com/asciimoo/hister/config"
	"github.com/asciimoo/hister/server/vectorstore"
)

// RerankedHit is one hit in the reranker's order.
type RerankedHit struct {
	DocID       string  `json:"doc_id"`
	URL         string  `json:"url"`
	RerankScore float64 `json:"rerank_score"`
}

// rerankCandidate is a hit as the reranker reads it: the title, then a
// passage. A semantic hit's passage is its best body chunk, a keyword hit's
// the start of its text.
type rerankCandidate struct {
	docID string
	url   string
	title string
	text  string
}

func keywordRerankCandidates(hits search.DocumentMatchCollection) []rerankCandidate {
	out := make([]rerankCandidate, 0, len(hits))
	for _, h := range hits {
		c := rerankCandidate{docID: h.ID}
		c.url, _ = h.Fields["url"].(string)
		c.title, _ = h.Fields["title"].(string)
		c.text, _ = h.Fields["text"].(string)
		out = append(out, c)
	}
	return out
}

// selectRerankCandidates takes the best keywordN keyword hits, fills up to
// total with the best semantic hits, then with further keyword hits if the
// semantic ones run out. A document in both lists is taken once.
func selectRerankCandidates(keyword, semantic []rerankCandidate, total, keywordN int) []rerankCandidate {
	out := make([]rerankCandidate, 0, total)
	seen := make(map[string]bool, total)
	take := func(list []rerankCandidate, limit int) []rerankCandidate {
		for len(list) > 0 && len(out) < limit {
			c := list[0]
			list = list[1:]
			if !seen[c.docID] {
				seen[c.docID] = true
				out = append(out, c)
			}
		}
		return list
	}
	keyword = take(keyword, min(keywordN, total))
	take(semantic, total)
	take(keyword, total)
	return out
}

func (c rerankCandidate) passage(maxChars int) string {
	text := c.text
	if c.title != "" {
		text = c.title + "\n" + text
	}
	if len(text) <= maxChars {
		return text
	}
	// Cut on a rune boundary.
	cut := maxChars
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut]
}

// rerank sets r.Reranked, or r.RerankError when the reranker fails. The
// results are otherwise left as they are, so a failure costs only the order.
func (i *Indexer) rerank(r *Results, cfg config.Rerank, query string, keyword, semantic []rerankCandidate) {
	candidates := selectRerankCandidates(keyword, semantic, cfg.Candidates, cfg.KeywordCandidates)
	if len(candidates) == 0 {
		return
	}
	passages := make([]string, len(candidates))
	for j, c := range candidates {
		passages[j] = c.passage(cfg.MaxDocumentChars)
	}
	start := time.Now()
	scores, err := vectorstore.NewReranker(&cfg).Rerank(context.Background(), query, passages)
	if err != nil {
		log.Warn().Err(err).Msg("rerank failed")
		r.RerankError = err.Error()
		return
	}
	log.Debug().Int("candidates", len(candidates)).Dur("took", time.Since(start)).Msg("reranked")
	r.Reranked = make([]RerankedHit, len(candidates))
	for j, c := range candidates {
		r.Reranked[j] = RerankedHit{DocID: c.docID, URL: c.url, RerankScore: scores[j]}
	}
	slices.SortStableFunc(r.Reranked, func(a, b RerankedHit) int {
		switch {
		case a.RerankScore > b.RerankScore:
			return -1
		case a.RerankScore < b.RerankScore:
			return 1
		}
		return 0
	})
}
