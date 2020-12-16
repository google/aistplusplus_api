"""TODO(ruilongli): DO NOT SUBMIT without one-line documentation for estimate_smpl.

TODO(ruilongli): DO NOT SUBMIT without a detailed description of estimate_smpl.
"""

from typing import Sequence

from absl import app


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
