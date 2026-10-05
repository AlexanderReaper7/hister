package querybuilder

import (
	"slices"
	"strings"

	"github.com/asciimoo/hister/server/indexer/searchschema"
)

// SearchExpression separates search directives from text matched by Bleve.
type SearchExpression struct {
	Text    string
	Sort    string
	HasSort bool
}

// ParseSearch separates supported search directives while preserving the
// remaining tokens for normal query building. Invalid directives remain
// searchable text.
func ParseSearch(input string) SearchExpression {
	expression := SearchExpression{Text: input}
	tokens, err := Tokenize(input)
	if err != nil {
		return expression
	}

	runes := []rune(input)
	parts := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if sortMode, ok := sortDirective(token); ok {
			expression.Sort = sortMode
			expression.HasSort = true
			continue
		}
		parts = append(parts, string(runes[token.start:token.end]))
	}
	if !expression.HasSort {
		return expression
	}

	expression.Text = strings.Join(parts, " ")
	if strings.TrimSpace(expression.Text) == "" {
		expression.Text = "*"
	}
	return expression
}

func sortDirective(token Token) (string, bool) {
	if token.Type != TokenWord {
		return "", false
	}
	value, ok := strings.CutPrefix(token.Value, "sort:")
	if !ok {
		return "", false
	}
	definition, ok := searchschema.LookupSort(value)
	if !ok {
		return "", false
	}
	if definition.Default {
		return "", true
	}
	return definition.Value, true
}

// SplitSemantic separates a query's filters from its free text. The text is
// what a semantic search embeds, so `type:web rust` embeds "rust" and not the
// filter. The filters, joined back into query syntax, decide which documents
// the semantic search may return.
//
// A filter is any token isFieldSpecific accepts, negated or not, and an
// alternation made only of such tokens. A negated plain word is not a filter:
// it stays in the text, as it did before filters reached semantic search.
func SplitSemantic(s string) (text, filters string) {
	tokens, err := Tokenize(s)
	if err != nil {
		return s, ""
	}
	runes := []rune(s)
	var textParts, filterParts []string
	for _, t := range tokens {
		if isFilterToken(t) {
			filterParts = append(filterParts, string(runes[t.start:t.end]))
		} else {
			textParts = append(textParts, string(runes[t.start:t.end]))
		}
	}
	return RemoveStandaloneWildcards(strings.Join(textParts, " ")), strings.Join(filterParts, " ")
}

func isFilterToken(t Token) bool {
	if t.Type == TokenAlternation {
		return len(t.Parts) > 0 && !slices.ContainsFunc(t.Parts, func(p Token) bool { return !isFieldSpecific(p) })
	}
	return isFieldSpecific(t)
}
