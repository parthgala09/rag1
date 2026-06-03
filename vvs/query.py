import argparse
import os
from pathlib import Path
from typing import Any, Dict

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
    from pinecone import Pinecone
except ImportError:
    raise SystemExit("Missing dependency: install pinecone with 'pip install pinecone'.")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_NAMESPACE = "default"


def load_env_vars() -> Dict[str, str]:
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", ""),
        "PINECONE_NAMESPACE": os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
    }
    missing = [
        name
        for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
        if not env[name]
    ]
    if missing:
        raise SystemExit("Missing required environment variables: " + ", ".join(missing))
    return env


def get_embedding(client: OpenAI, text: str):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        encoding_format="float",
    )
    return response.data[0].embedding


def build_filter(args: argparse.Namespace) -> Dict[str, Any]:
    filter_payload: Dict[str, Any] = {}
    if args.year:
        filter_payload["year"] = {"$eq": args.year}
    if args.doc_type:
        filter_payload["doc_type"] = {"$eq": args.doc_type}
    if args.doc_id:
        filter_payload["doc_id"] = {"$eq": args.doc_id}
    if args.page_number is not None:
        filter_payload["page_number"] = {"$eq": args.page_number}
    return filter_payload


def validate_index_dimension(pc: Pinecone, index_name: str) -> None:
    description = pc.describe_index(index_name)
    existing_dimension = getattr(description, "dimension", None)
    if existing_dimension != EMBEDDING_DIMENSIONS:
        raise SystemExit(
            f"Pinecone index '{index_name}' has dimension {existing_dimension}, "
            f"but {EMBEDDING_MODEL} requires {EMBEDDING_DIMENSIONS}. "
            "Use a 1536-dimensional index or change PINECONE_INDEX_NAME to a new index name."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a Pinecone vector index with optional metadata filters."
    )
    parser.add_argument("--query", required=True, help="The question or search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top chunks to retrieve.")
    parser.add_argument("--year", help="Optional year filter, for example 2025.")
    parser.add_argument("--doc-type", dest="doc_type", help="Optional document type filter.")
    parser.add_argument("--doc-id", dest="doc_id", help="Optional document id filter.")
    parser.add_argument("--page-number", dest="page_number", type=int, help="Optional page filter.")
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional Pinecone namespace override. Defaults to PINECONE_NAMESPACE or default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_env_vars()

    openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])
    pc = Pinecone(api_key=env["PINECONE_API_KEY"])
    validate_index_dimension(pc, env["PINECONE_INDEX_NAME"])
    index = pc.Index(env["PINECONE_INDEX_NAME"])
    namespace = args.namespace or env["PINECONE_NAMESPACE"]

    query_embedding = get_embedding(openai_client, args.query)
    filter_payload = build_filter(args)

    response = index.query(
        vector=query_embedding,
        top_k=args.top_k,
        include_metadata=True,
        include_values=False,
        filter=filter_payload if filter_payload else None,
        namespace=namespace,
    )

    print(f"Query: {args.query}")
    print(f"Namespace: {namespace}")
    if filter_payload:
        print(f"Filters: {filter_payload}")
    print(f"Retrieved {len(response.matches)} results")
    print("---")

    for rank, match in enumerate(response.matches, start=1):
        metadata = match.metadata or {}
        snippet = metadata.get("text", "(no text metadata)")
        snippet = snippet[:700].replace("\n", " ")
        print(f"Rank {rank}: id={match.id} score={match.score}")
        print(
            "  "
            f"doc_id={metadata.get('doc_id')} "
            f"title={metadata.get('title')} "
            f"year={metadata.get('year')} "
            f"doc_type={metadata.get('doc_type')} "
            f"page_number={metadata.get('page_number')}"
        )
        print(f"  snippet: {snippet}")
        print("---")


if __name__ == "__main__":
    main()
