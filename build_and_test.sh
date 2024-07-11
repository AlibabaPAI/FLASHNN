#!/bin/bash
python3 setup.py develop
python3 tests/run_tests.py --test unit
python3 -m pip uninstall flashnn -y
rm -fr dist
python3 ./setup.py bdist_wheel
python3 -m pip install dist/*.whl
