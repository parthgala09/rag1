# Pinecone PDF Vector Search Setup

This folder indexes a 200+ page PDF corpus with OpenAI embeddings and Pinecone DB, then queries the indexed chunks with optional metadata filters.

## Environment

Create `vvs/.env` with:

```text
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-pinecone-index-name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=default
PINECONE_METRIC=cosine
```

`PINECONE_CLOUD`, `PINECONE_REGION`, `PINECONE_NAMESPACE`, and `PINECONE_METRIC` are optional. They default to `aws`, `us-east-1`, `default`, and `cosine`.

## Dependencies

```bash
pip install -r vvs/requirements.txt
```

## Index Config

- Corpus: `vvs/30th Annual Report.pdf`
- Corpus source: NSE Clearing Limited Thirtieth Annual Report 2024-25
- Pages: 201
- Embedding model: `text-embedding-3-small`
- Embedding dimensions: `1536`
- Vector database: Pinecone dense index
- Distance metric: `cosine`
- Chunk size: `512` tokens
- Chunk overlap: `64` tokens
- Metadata: `doc_id`, `title`, `year`, `doc_type`, `page_number`, `chunk_index`, `source_file`, `text`

If the Pinecone index named by `PINECONE_INDEX_NAME` does not exist, `ingest.py` creates a serverless dense index with the config above.

If that index already exists, it must have dimension `1536`. A 512-dimensional index cannot store `text-embedding-3-small` vectors; set `PINECONE_INDEX_NAME` to a new index name or recreate the existing index with dimension `1536`.

## Ingest

Verify parsing and chunk counts without API calls:

```bash
python vvs/ingest.py --pdf-dir vvs --dry-run
```

Ingest and upload vectors:

```bash
python vvs/ingest.py --pdf-dir vvs --manifest vvs/corpus_manifest.csv
```

The manifest is written to `vvs/corpus_manifest.csv` with `doc_id`, `title`, `year`, `doc_type`, `num_pages`, and `num_chunks`.

## Query

Run semantic search:

```bash
python vvs/query.py --query "What are the financial highlights?" --top-k 5
```

Run filtered semantic search:

```bash
python vvs/query.py --query "What does the report say about risk management?" --year 2025 --doc-type annual_report --top-k 5
```

Supported filters:

- `--year`
- `--doc-type`
- `--doc-id`
- `--page-number`
- `--namespace`

## Example Workflow

```bash
pip install -r vvs/requirements.txt
python3 ingest.py --pdf-dir vvs --dry-run
python3 ingest.py --pdf-dir vvs --manifest vvs/corpus_manifest.csv
python3 query.py --query "What are the audit committee findings?" --doc-type annual_report --top-k 5
```
