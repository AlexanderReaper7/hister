package querybuilder

import (
	"testing"

	"github.com/blevesearch/bleve/v2"
)

func TestConvenienceFiltersCannotBeSatisfiedByQueryText(t *testing.T) {
	for _, input := range []string{"needle site:example.com", "needle has:label"} {
		t.Run(input, func(t *testing.T) {
			mapping := bleve.NewIndexMapping()
			for _, name := range []string{"domain", "label"} {
				field := bleve.NewTextFieldMapping()
				field.Analyzer = "keyword"
				mapping.DefaultMapping.AddFieldMappingsAt(name, field)
			}
			idx, err := bleve.NewMemOnly(mapping)
			if err != nil {
				t.Fatal(err)
			}
			defer idx.Close()
			for id, doc := range map[string]map[string]string{
				"selected": {"title": "needle", "domain": "example.com", "label": "research"},
				"literal":  {"title": input, "domain": "other.test", "label": ""},
			} {
				if err := idx.Index(id, doc); err != nil {
					t.Fatal(err)
				}
			}
			q, err := BuildValidated(input)
			if err != nil {
				t.Fatal(err)
			}
			result, err := idx.Search(bleve.NewSearchRequest(q))
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Hits) != 1 || result.Hits[0].ID != "selected" {
				t.Fatalf("Search(%q) = %v, want only selected document", input, result.Hits)
			}
		})
	}
}
