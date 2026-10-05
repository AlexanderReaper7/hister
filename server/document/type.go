package document

// DocType represents the type of an indexed document.
type DocType int

const (
	Web DocType = iota
	Local
	RemoteFile
	// Code is a piece of a source file submitted by an external code indexer,
	// such as semsearch. Its URL opens the file at the piece's first line.
	Code
)

// String returns the human readable name of the DocType.
func (t DocType) String() string {
	switch t {
	case Web:
		return "web"
	case Local:
		return "local"
	case RemoteFile:
		return "remote"
	case Code:
		return "code"
	default:
		return "unknown"
	}
}
