import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Missing dependency: install openai with 'pip install openai'.")

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None
    ServerlessSpec = None

try:
    from pypdf import PdfReader
except ImportError:
    raise SystemExit("Missing dependency: install pypdf with 'pip install pypdf'.")

try:
    import tiktoken
except ImportError:
    raise SystemExit("Missing dependency: install tiktoken with 'pip install tiktoken'.")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_METRIC = "cosine"
DEFAULT_NAMESPACE = "default"


def load_env_vars(require_api: bool = True) -> Dict[str, str]:
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", ""),
        "PINECONE_CLOUD": os.getenv("PINECONE_CLOUD", "aws"),
        "PINECONE_REGION": os.getenv("PINECONE_REGION", "us-east-1"),
        "PINECONE_METRIC": os.getenv("PINECONE_METRIC", DEFAULT_METRIC),
        "PINECONE_NAMESPACE": os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
    }
    if require_api:
        missing = [
            name
            for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
            if not env[name]
        ]
        if missing:
            raise SystemExit("Missing required environment variables: " + ", ".join(missing))
    return env


def get_token_encoder() -> "tiktoken.Encoding":
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


def chunk_text(text: str, max_tokens: int, overlap: int) -> Iterable[str]:
    if overlap >= max_tokens:
        raise ValueError("--overlap must be smaller than --max-tokens")
    if not text.strip():
        return

    encoding = get_token_encoder()
    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        yield text.strip()
        return

    step = max_tokens - overlap
    for start in range(0, len(token_ids), step):
        chunk_tokens = token_ids[start : start + max_tokens]
        chunk = encoding.decode(chunk_tokens).strip()
        if chunk:
            yield chunk
        if start + max_tokens >= len(token_ids):
            break


def filename_to_metadata(file_path: Path) -> Dict[str, str]:
    name = file_path.stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

    if slug == "30th_annual_report":
        return {
            "doc_id": "30th_annual_report",
            "title": "NSE Clearing Limited Thirtieth Annual Report 2024-25",
            "year": "2025",
            "doc_type": "annual_report",
        }

    year_match = re.search(r"(19|20)\d{2}", name)
    lower = name.lower()
    if "annual" in lower and "report" in lower:
        doc_type = "annual_report"
    elif "circular" in lower:
        doc_type = "circular"
    elif "summary" in lower:
        doc_type = "summary"
    elif "paper" in lower:
        doc_type = "paper"
    else:
        doc_type = "report"

    return {
        "doc_id": slug or name,
        "title": name.replace("_", " ").replace("-", " ").strip(),
        "year": year_match.group(0) if year_match else "",
        "doc_type": doc_type,
    }


def parse_pdf(file_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(file_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_number, text.strip()))
    return pages


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
):
    existing_names = pc.list_indexes().names()
    if index_name not in existing_names:
        print(
            f"Creating Pinecone index '{index_name}' "
            f"with dimension={dimension}, metric={metric}, cloud={cloud}, region={region}"
        )
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pc.describe_index(index_name).status["ready"]:
            print("Waiting for Pinecone index to be ready...")
            time.sleep(5)
    else:
        description = pc.describe_index(index_name)
        existing_dimension = getattr(description, "dimension", None)
        existing_metric = getattr(description, "metric", None)
        if existing_dimension != dimension:
            raise SystemExit(
                f"Pinecone index '{index_name}' has dimension {existing_dimension}, "
                f"but {EMBEDDING_MODEL} requires {dimension}. "
                "Use a 1536-dimensional index or change PINECONE_INDEX_NAME to a new index name."
            )
        if existing_metric and existing_metric != metric:
            raise SystemExit(
                f"Pinecone index '{index_name}' uses metric {existing_metric}, "
                f"but this project is configured for {metric}."
            )
    return pc.Index(index_name)


def upsert_batch(index, records: List[Dict], namespace: str) -> None:
    if records:
        index.upsert(vectors=records, namespace=namespace)


def build_chunk_records(
    pdf_path: Path,
    max_tokens: int,
    overlap: int,
) -> Tuple[Dict[str, str], int, List[Dict[str, object]]]:
    doc_meta = filename_to_metadata(pdf_path)
    page_data = parse_pdf(pdf_path)
    records = []

    for page_number, text in page_data:
        if not text:
            continue
        for chunk_index, chunk in enumerate(chunk_text(text, max_tokens, overlap), start=1):
            chunk_id = f"{doc_meta['doc_id']}_{page_number}_{chunk_index}"
            records.append(
                {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_meta["doc_id"],
                        "title": doc_meta["title"],
                        "year": doc_meta["year"],
                        "doc_type": doc_meta["doc_type"],
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "source_file": pdf_path.name,
                        "text": chunk,
                    },
                }
            )

    return doc_meta, len(page_data), records


def ingest_pdf(
    pdf_path: Path,
    openai_client: OpenAI,
    index,
    namespace: str,
    max_tokens: int,
    overlap: int,
    batch_size: int,
    dry_run: bool,
) -> Dict[str, int]:
    doc_meta, num_pages, records = build_chunk_records(pdf_path, max_tokens, overlap)
    print(
        f"Processing {pdf_path.name}: doc_id={doc_meta['doc_id']} "
        f"title={doc_meta['title']} chunks={len(records)}"
    )

    if dry_run:
        return {"num_pages": num_pages, "num_chunks": len(records)}

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        embeddings = embed_texts(openai_client, [record["text"] for record in batch])
        vectors = [
            {
                "id": record["id"],
                "values": embedding,
                "metadata": record["metadata"],
            }
            for record, embedding in zip(batch, embeddings)
        ]
        upsert_batch(index, vectors, namespace)
        print(f"Upserted {min(start + batch_size, len(records))}/{len(records)} chunks")

    return {"num_pages": num_pages, "num_chunks": len(records)}


def write_manifest(rows: List[Dict[str, object]], manifest_path: Path) -> None:
    fieldnames = ["doc_id", "title", "year", "doc_type", "num_pages", "num_chunks"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_pdf_paths(pdf_dir: Path) -> List[Path]:
    return sorted(pdf_dir.rglob("*.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into Pinecone using OpenAI text-embedding-3-small."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing PDF files to ingest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("corpus_manifest.csv"),
        help="Output manifest CSV path.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk PDFs, then write the manifest without embedding or uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env_vars(require_api=not args.dry_run)

    openai_client: Optional[OpenAI] = None
    index = None
    if not args.dry_run:
        if Pinecone is None or ServerlessSpec is None:
            raise SystemExit("Missing dependency: install pinecone with 'pip install pinecone'.")
        openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
        pc = Pinecone(api_key=env["PINECONE_API_KEY"])
        index = ensure_index(
            pc=pc,
            index_name=env["PINECONE_INDEX_NAME"],
            dimension=EMBEDDING_DIMENSIONS,
            metric=env["PINECONE_METRIC"],
            cloud=env["PINECONE_CLOUD"],
            region=env["PINECONE_REGION"],
        )

    pdf_paths = collect_pdf_paths(args.pdf_dir)
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {args.pdf_dir}")

    manifest_rows = []
    for pdf_path in pdf_paths:
        stats = ingest_pdf(
            pdf_path=pdf_path,
            openai_client=openai_client,
            index=index,
            namespace=env["PINECONE_NAMESPACE"],
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        doc_meta = filename_to_metadata(pdf_path)
        manifest_rows.append(
            {
                "doc_id": doc_meta["doc_id"],
                "title": doc_meta["title"],
                "year": doc_meta["year"],
                "doc_type": doc_meta["doc_type"],
                "num_pages": stats["num_pages"],
                "num_chunks": stats["num_chunks"],
            }
        )

    write_manifest(manifest_rows, args.manifest)
    mode = "Dry run finished" if args.dry_run else "Ingestion finished"
    print(f"{mode}. Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
