"""Utilisation en ligne de commande.

    python cli.py add docs/cours.pdf docs/contrat.pdf
    python cli.py ask "Quelle est la date limite ?"
    python cli.py chat
    python cli.py reset
"""

import argparse

from rag import MiniRAG


def print_answer(answer, hits):
    print("\n" + answer + "\n")
    print("Sources :")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h['source']} p.{h['page']} (score {h['score']})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Pose des questions à tes PDF.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="indexer un ou plusieurs PDF")
    p_add.add_argument("files", nargs="+")

    p_ask = sub.add_parser("ask", help="poser une question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4)

    sub.add_parser("chat", help="mode conversation")
    sub.add_parser("list", help="lister les documents indexés")
    sub.add_parser("reset", help="vider la base")

    args = parser.parse_args()
    rag = MiniRAG()

    if args.cmd == "add":
        for f in args.files:
            n = rag.add_pdf(f)
            print(f"{f} : {n} morceaux indexés")
    elif args.cmd == "ask":
        print_answer(*rag.ask(args.question, args.k))
    elif args.cmd == "chat":
        print("Tape ta question (ou 'q' pour quitter).")
        while True:
            q = input("> ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                print_answer(*rag.ask(q))
    elif args.cmd == "list":
        for s in rag.sources():
            print("-", s)
    elif args.cmd == "reset":
        rag.reset()
        print("Base vidée.")


if __name__ == "__main__":
    main()
