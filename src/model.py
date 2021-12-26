from models.parameter import PARAMETERS
import argparse
import re

parser = argparse.ArgumentParser(
    description='Print a list of models.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('regex', nargs='*', type=str, help='Regex of dataset name or model name.')


def main() -> None:
    args = parser.parse_args()
    regexes = [re.compile(r) for r in args.regex]

    # list of models
    model_names = []

    for model_name in PARAMETERS.keys():
        match = True

        for regex in regexes:
            match &= regex.search(model_name) is not None

        if match:
            model_names.append(model_name)

    model_names.sort()

    # print models
    for model_name in model_names:
        print(model_name)


if __name__ == '__main__':
    main()
