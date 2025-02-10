#!/bin/bash

./main.py <(sed "s#{{WORKING_DIRECTORY}}#$(pwd)#" test/config.yaml)
