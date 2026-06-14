#!/usr/bin/env python3
import argparse
from pathlib import Path


BLOCK = """\
Section {i}
The benchmark text is deterministic and intentionally repetitive. It contains
code, data, and prose so the tokenizer sees a realistic long prompt.

```cpp
int mcd_value_{i} = {i};
for (int j = 0; j < 8; ++j) {{
    mcd_value_{i} += j;
}}
```

Facts:
- Windows CUDA performs prompt prefill.
- Mac Metal performs local decode.
- Direct 10GbE carries only binary state payloads.

"""


def generate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        text = []
        i = 0
        # This is a character target only. Actual token counts are measured
        # with llama-server /tokenize after selecting a model.
        while len("".join(text)) < target * args.chars_per_token:
            text.append(BLOCK.format(i=i))
            i += 1
        path = out / f"prompt_{target}.txt"
        path.write_text("".join(text), encoding="utf-8")
        print(f"{path} chars={path.stat().st_size}")


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic MCD prompts")
    parser.add_argument("--out", default="results/mcd/prompts")
    parser.add_argument("--targets", type=int, nargs="+", default=[512, 2048, 8192, 16384, 32768])
    parser.add_argument("--chars-per-token", type=int, default=4)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
