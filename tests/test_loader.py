"""TODO(ruilongli): DO NOT SUBMIT without one-line documentation for test_loader.

TODO(ruilongli): DO NOT SUBMIT without a detailed description of test_loader.
"""

from typing import Sequence

from absl import app


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
