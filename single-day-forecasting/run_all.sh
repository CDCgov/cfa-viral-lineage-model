#!/bin/sh

for conf in `ls config/`; do ./main.py config/$conf; done
