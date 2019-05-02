#!/usr/bin/env bash
set -eux
#python dqn.py --env Freeway-v0 --steps 100000
python run.py --env Boxing-v0 --steps 100000
