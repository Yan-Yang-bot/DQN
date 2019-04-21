#!/usr/bin/env bash
set -eux
python dqn.py --env Freeway-v0 --steps 10000
#python run.py --env Boxing-v0
