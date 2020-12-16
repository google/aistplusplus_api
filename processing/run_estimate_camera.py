"""TODO(ruilongli): DO NOT SUBMIT without one-line documentation for estimate_camera.

TODO(ruilongli): DO NOT SUBMIT without a detailed description of estimate_camera.
"""

from typing import Sequence

from absl import app


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
