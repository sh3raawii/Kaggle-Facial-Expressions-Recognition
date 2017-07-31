#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for file in $DIR/Training/*; do rm -rf "$file"; done
for file in $DIR/PublicTest/*; do rm -rf "$file"; done
for file in $DIR/PrivateTest/*; do rm -rf "$file"; done
