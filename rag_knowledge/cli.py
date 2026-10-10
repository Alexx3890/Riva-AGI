"""CLI for rag_knowledge."""

import argparse
import asyncio
import os
import sys

from . import load_env
from .retrieval.retriever import RetrievalError
from .service import get_rag_service, query_rag


def list_knowledge_entries():
    """List registered knowledge entries."""
    service = get_rag_service()
    try:
        docs = service.retriever.documents
    except Exception as e:
        print(f"\n[Error] Unable to list knowledge documents (database outage/connection failure): {e}\n")
        return
    total_str = f"{len(docs)}" if len(docs) < 100 else f"{len(docs)}+ (limit 100 reached)"
    print(f"\n--- Registered Knowledge Documents ({total_str}) ---")
    for doc in docs:
        print(f" * [{doc.get('id')}] {doc.get('title')}")
        print(f"   Keywords: {', '.join(doc.get('keywords', []))}")
        print(f"   Summary:  {doc.get('summary')}")
        print()


async def run_query(query: str, verbose: bool = True):
    """Execute a query against the RAG service and display results."""
    service = get_rag_service()
    pre_matches = None
    cli_top_k = int(os.getenv("RAG_TOP_K", "10"))

    if verbose:
        try:
            pre_matches = await asyncio.to_thread(service.retriever.retrieve, query, top_k=cli_top_k)
            print(f"\n[Retrieval Matches for '{query}']:")
            if pre_matches:
                for idx, m in enumerate(pre_matches, 1):
                    meta = m.get("metadata") or {}
                    src = meta.get("source") or meta.get("source_file") or ""
                    pg = meta.get("page")
                    src_str = f" ({src}" + (f", p. {pg}" if pg is not None else "") + ")" if src else ""
                    print(f"  {idx}. {m.get('title')}{src_str} (score={m.get('score')})")
            else:
                print("  (No documents met the relevance threshold)")
        except RetrievalError as e:
            print(f"\n[Outage Error]: Knowledge database is unreachable or encountered an outage:\n  {e}")
            print("\n[Status]: Database failure detected. Please verify Qdrant connection and service health.\n")
            return
        except Exception as e:
            print(f"\n[Unexpected Error]: Retrieval failed:\n  {e}\n")
            return

    print(f"\n[Query]: {query}")
    try:
        answer, sources = await service.query_with_sources(query, pre_retrieved=pre_matches)
        print(f"\n[Answer]:\n{answer}\n")
        if sources:
            print("[Sources]:")
            for s in sources:
                print(f"  - {s}")
            print()
    except RetrievalError as e:
        print(f"\n[Outage Error]: Knowledge database is unreachable or encountered an outage:\n  {e}")
        print("\n[Status]: Database failure detected. Please verify Qdrant connection and service health.\n")
    except Exception as e:
        print(f"\n[Error]: Failed to synthesize answer:\n  {e}\n")


def main():
    load_env()
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base - Standalone CLI & Retrieval Tool"
    )
    parser.add_argument(
        "subcommand_or_query",
        nargs="?",
        default=None,
        help="Command ('ask', 'list') or search query string (e.g. 'What events are scheduled?')",
    )
    parser.add_argument(
        "ask_query",
        nargs="?",
        default=None,
        help="Query string if 'ask' subcommand is used (e.g. 'python -m rag_knowledge ask ...')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all documents in the knowledge base",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print retrieval scoring and matching details (enabled by default)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress retrieval ranking matches and print only the final answer",
    )

    args = parser.parse_args()

    if args.list or args.subcommand_or_query == "list":
        list_knowledge_entries()
        return

    # Determine query string whether invoked as `ask <query>` or `<query>`
    if args.subcommand_or_query == "ask":
        query_text = args.ask_query
    else:
        query_text = args.subcommand_or_query

    if not query_text:
        parser.print_help()
        sys.exit(1)

    show_matches = False if args.quiet else True
    asyncio.run(run_query(query_text, verbose=show_matches))


if __name__ == "__main__":
    main()
