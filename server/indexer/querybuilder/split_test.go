// SPDX-License-Identifier: AGPL-3.0-or-later

package querybuilder

import "testing"

func TestSplitSemantic(t *testing.T) {
	for _, tc := range []struct {
		input, text, filters string
	}{
		{"rust async", "rust async", ""},
		{"type:web rust", "rust", "type:web"},
		{"rust -domain:example.com", "rust", "-domain:example.com"},
		{"site:github.com updated:>2w async runtime", "async runtime", "site:github.com updated:>2w"},
		{`title:"exact phrase" rust`, "rust", `title:"exact phrase"`},
		{"(domain:a.example|domain:b.example) rust", "rust", "(domain:a.example|domain:b.example)"},
		{"(rust|go) type:web", "rust|go", "type:web"},
		{"metadata.visits:3 rust", "rust", "metadata.visits:3"},
		{"-rust go", "-rust go", ""},
		{"* type:web", "", "type:web"},
		{"updated:yesterday rust", "updated:yesterday rust", ""},
	} {
		text, filters := SplitSemantic(tc.input)
		if text != tc.text || filters != tc.filters {
			t.Errorf("SplitSemantic(%q) = %q, %q, want %q, %q", tc.input, text, filters, tc.text, tc.filters)
		}
	}
}
