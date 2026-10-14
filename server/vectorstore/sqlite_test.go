// SPDX-License-Identifier: AGPL-3.0-or-later

package vectorstore

import (
	"database/sql"
	"database/sql/driver"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/asciimoo/hister/server/vectorstore/sqlitevec"

	sqlite3 "github.com/mattn/go-sqlite3"
)

// vec0 starts the idxStr of a plan that reads every vector with '1'
// (VEC0_QUERY_PLAN_FULLSCAN in sqlite-vec.c).
var vec0FullScan = regexp.MustCompile(`SCAN embeddings VIRTUAL TABLE INDEX \d+:1`)

// recordingDriver opens go-sqlite3 connections that record each statement
// prepared on them. Its connections offer only Prepare, so database/sql
// prepares every Exec and Query and each one is recorded.
type recordingDriver struct {
	mu      sync.Mutex
	queries []string
}

func (d *recordingDriver) Open(name string) (driver.Conn, error) {
	conn, err := (&sqlite3.SQLiteDriver{}).Open(name)
	if err != nil {
		return nil, err
	}
	return recordingConn{Conn: conn, driver: d}, nil
}

func (d *recordingDriver) take() []string {
	d.mu.Lock()
	defer d.mu.Unlock()
	queries := d.queries
	d.queries = nil
	return queries
}

type recordingConn struct {
	driver.Conn
	driver *recordingDriver
}

func (c recordingConn) Prepare(query string) (driver.Stmt, error) {
	c.driver.mu.Lock()
	c.driver.queries = append(c.driver.queries, query)
	c.driver.mu.Unlock()
	return c.Conn.Prepare(query)
}

var (
	recording         = &recordingDriver{}
	registerRecording sync.Once
)

func TestSQLiteDeleteLooksUpEmbeddingsByKey(t *testing.T) {
	sqlitevec.Auto()
	registerRecording.Do(func() { sql.Register("sqlite3_recording", recording) })
	path := filepath.Join(t.TempDir(), "vectors.sqlite3")
	db, err := sql.Open("sqlite3_recording", path)
	if err != nil {
		t.Fatalf("open vector database: %v", err)
	}
	db.SetMaxOpenConns(1)
	store := &sqliteVectorStore{db: db, dimensions: 2}
	defer store.Close() //nolint:errcheck
	if err := store.Init(); err != nil {
		t.Fatalf("Init() error: %v", err)
	}
	chunks := []Chunk{
		{Index: 0, Text: "first", Embedding: []float32{1, 0}},
		{Index: 1, Text: "second", Embedding: []float32{0, 1}},
	}
	for _, doc := range []string{"doc-a", "doc-b"} {
		if err := store.PutChunks(doc, 0, chunks); err != nil {
			t.Fatalf("PutChunks(%s) error: %v", doc, err)
		}
	}
	recording.take()

	// Re-embedding a document and deleting it both remove its old chunks.
	if err := store.PutChunks("doc-a", 0, chunks); err != nil {
		t.Fatalf("PutChunks(doc-a) again error: %v", err)
	}
	if err := store.Delete("doc-a"); err != nil {
		t.Fatalf("Delete(doc-a) error: %v", err)
	}
	if err := store.Delete("doc-missing"); err != nil {
		t.Fatalf("Delete(doc-missing) error: %v", err)
	}

	plain, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatalf("open plain connection: %v", err)
	}
	defer plain.Close() //nolint:errcheck
	checked := 0
	for _, query := range recording.take() {
		if !strings.HasPrefix(strings.TrimSpace(query), "DELETE FROM embeddings") {
			continue
		}
		checked++
		// The plan depends on which columns are constrained, not on the values.
		args := make([]any, strings.Count(query, "?"))
		rows, err := plain.Query("EXPLAIN QUERY PLAN "+query, args...)
		if err != nil {
			t.Fatalf("explain %q: %v", query, err)
		}
		for rows.Next() {
			var id, parent, unused int
			var detail string
			if err := rows.Scan(&id, &parent, &unused, &detail); err != nil {
				t.Fatalf("scan plan of %q: %v", query, err)
			}
			if vec0FullScan.MatchString(detail) {
				t.Errorf("%s\nplans as %q, a scan of every vector", query, detail)
			}
		}
		if err := rows.Close(); err != nil {
			t.Fatalf("close plan of %q: %v", query, err)
		}
	}
	if checked == 0 {
		t.Fatal("no DELETE on embeddings was recorded")
	}

	var vectors, docA int
	if err := plain.QueryRow(`SELECT count(*) FROM embeddings_rowids`).Scan(&vectors); err != nil {
		t.Fatalf("count vectors: %v", err)
	}
	if err := plain.QueryRow(`SELECT count(*) FROM chunk_meta WHERE doc_id = 'doc-a'`).Scan(&docA); err != nil {
		t.Fatalf("count doc-a chunks: %v", err)
	}
	if vectors != len(chunks) || docA != 0 {
		t.Fatalf("after deleting doc-a: %d vectors, %d doc-a chunks; want %d and 0", vectors, docA, len(chunks))
	}
}
