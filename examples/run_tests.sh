#!/bin/sh

set -e

# Run examples and tests to make sure they finish.
for t in 0_*.py 1_*.py 1[bcdefgi]*.py test_lazy_adam.py; do
  echo "Running $t"
  python3 "$t" --threads=4 --epochs=2
  echo
done
python3 test_loggers.py --fs_logger --tb_logger
