#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

def main():
    try:
        print("123四五六")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()