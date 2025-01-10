import argparse

from defgsus import training, testing, show
from defgsus.config import EXPERIMENTS


def main():
    main_parser = argparse.ArgumentParser()

    subparsers = main_parser.add_subparsers()

    parser = subparsers.add_parser("train", help="Run all unfinished trainings")
    parser.set_defaults(command="train")
    parser.add_argument(
        "-r", "--reset", type=bool, nargs="?", default=False, const=True,
        help="Run existing experiment again"
    )

    parser = subparsers.add_parser("test", help="Run all unfinished tests")
    parser.set_defaults(command="test")

    parser = subparsers.add_parser("list", help="List all experiments and their hash-codes")
    parser.set_defaults(command="list")

    parser = subparsers.add_parser("table", help="Show table of finished experiments")
    parser.set_defaults(command="table")
    parser.add_argument(
        "-t", "--tests", type=bool, nargs="?", default=False, const=True,
        help="Include finished test results"
    )
    parser.add_argument(
        "-s", "--slugs", type=bool, nargs="?", default=False, const=True,
        help="Include model links"
    )
    parser.add_argument(
        "-m", "--markdown", type=bool, nargs="?", default=False, const=True,
        help="Include markdown-compatible html elements"
    )

    parser = subparsers.add_parser("test-table", help="Show table of finished tests")
    parser.set_defaults(command="test-table")

    parser = subparsers.add_parser("markdown", help="Show table and markdown output of finished experiments")
    parser.set_defaults(command="markdown")
    parser.add_argument(
        "-t", "--tests", type=bool, nargs="?", default=False, const=True,
        help="Include finished test results"
    )

    parser = subparsers.add_parser("reconstructions", help="Render reconstructions of trained models")
    parser.set_defaults(command="reconstructions")

    args = main_parser.parse_args()

    if args.command == "train":
        training.train_experiments(EXPERIMENTS, reset=args.reset)
        show.show_experiments_table(EXPERIMENTS)

    elif args.command == "test":
        testing.test_experiments(EXPERIMENTS)

    elif args.command == "list":
        show.list_experiments(EXPERIMENTS)

    elif args.command == "table":
        show.show_experiments_table(
            EXPERIMENTS,
            tests=args.tests,
            slugs=args.slugs,
            markdown=args.markdown,
        )

    elif args.command == "test-table":
        show.show_test_table(EXPERIMENTS)

    elif args.command == "markdown":
        show.show_experiments_table(EXPERIMENTS, tests=args.tests, slugs=True, markdown=True)
        show.show_markdown(EXPERIMENTS)

    elif args.command == "reconstructions":
        show.render_reconstructions(EXPERIMENTS)


if __name__ == "__main__":
    main()
