#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Script to convert structured output schema generation jsonl file to jsonl evaluation format.

Example command:

python convert_to_eval_jsonl.py \
--input_dir='data/labels' \
--output_file=data/so_bench_eval.jsonl \
--num_threads=16
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
import base64
import hashlib
import glob
import uuid
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image


Schema = Dict[str, Any]

_SYSTEM_PROMPT = """# SYSTEM PROMPT — Visual Structured Information Extractor

You are an expert at extracting structured information from images according to provided schemas.
Given the image and user intent, extract information that matches the provided json schema.

## Constraint-Aware Default Rules (for missing/unextractable required fields)
**General precedence**
- **P0 — Use explicit schema defaults**: If provided in schema, prefer these.
- **P1 — Respect enumerations/const**: Use the declared `const` or the first `enum` value (in stable order).
- **P2 — Apply type-specific rules while respecting schema constraints**.

**Strings**
- Default placeholder: `"#"`.
- Respect `minLength`/`maxLength`:
• If `minLength = m ≥ 1`, return exactly `m` hash characters.
• If `minLength = maxLength = n`, return `n` hash characters.
- If `pattern` forbids `#`, construct the shortest valid alternative using allowed characters (`"A"`, `"0"`, etc.).
- Only output `null` if `"null"` is explicitly allowed.

**Integers & Numbers**
- Default to `0`.
- If `minimum`/`exclusiveMinimum` is present, choose the **lowest valid** value.
- If `maximum`/`exclusiveMaximum` is present and `0` violates it, choose the **highest valid** value under the bound.
- If `multipleOf` is present, adjust to the nearest valid multiple.

**Booleans**
- Default to `false` unless otherwise constrained.

**Dates / Times**
- Use RFC 3339-valid placeholders:
• `"1970-01-01"` for `date`
• `"1970-01-01T00:00:00Z"` for `date-time`
• `"00:00:00Z"` for `time`
- If stricter patterns exist, choose the simplest valid match.

**Arrays**
- If `minItems = k`, return exactly `k` items, each filled recursively with these same rules.
- Ensure uniqueness if `uniqueItems: true`.

**Objects**
- Populate all required fields recursively.
- Never add undeclared fields.

---

## Edge-Case Notes
- Never hallucinate values. Required fields must be filled with constraint-compliant placeholders if unextractable.
- Optional fields may be omitted or set to `null` (if schema allows).
- If constraints are contradictory, return the closest minimally violating placeholder.
- Stable tie-breaking: when multiple placeholder choices are possible, use the first in schema order.

---

## Output Format
- Return **only valid JSON**, wrapped in a fenced markdown block:
```json
{ ... }
```
- All required fields must appear.
- Placeholders must respect schema constraints.
- No comments, no extra text, no extra fields.
"""

_USER_PROMPT = """JSON Schema:
```json
{schema_json}
```

User Intent: {user_intent}

Please analyze the image and extract structured information according to the schema above, honoring the user intent.
Return only valid JSON in a fenced ```json code block."""


def generate_id(example: Dict[str, Any]) -> str:
    """Generate a UUID from hashing the example.

    Args:
        example: Dictionary containing example data to hash

    Returns:
        A 32-character hexadecimal string ID generated from the example hash
    """
    json_string = json.dumps(example, sort_keys=True)
    hash_object = hashlib.sha256(json_string.encode("utf-8"))
    hex_digest = hash_object.hexdigest()
    generated_uuid = uuid.UUID(hex_digest[:32])
    ex_id = generated_uuid.hex[:32]  # Take the first 32 characters
    return ex_id


def encode_image_from_path(image_path: str) -> str:
    """Encode image from file path to base64 with optional size optimization.

    Args:
        image_path: Path to the image file (local or remote)

    Returns:
        Base64 encoded image string
    """
    with Image.open(Path(image_path)) as img:
        rgb_img = img.convert("RGB")

        buffer = BytesIO()
        rgb_img.save(buffer, format="JPEG", quality=95)
        jpeg_bytes = buffer.getvalue()

    # Encode the JPEG bytes into a Base64 string
    return base64.b64encode(jpeg_bytes).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--first_n", type=int, default=-1)

    args = parser.parse_args()
    input_data = []
    files = glob.glob(f"{args.input_dir}/*.json")
    print(f"Found {len(files)} files")
    if args.first_n > 0:
        files = files[: args.first_n]

    def load_one(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data

    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        input_data = list(ex.map(load_one, files))

    with open(args.output_file, "w") as f:
        for _, example in tqdm(enumerate(input_data)):
            image_url = example["image_url"]
            try:
                base64_image = encode_image_from_path(image_url)
                schema = example["schema_json"]
                messages = [
                    {
                        "role": "system",
                        "content": _SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _USER_PROMPT.format(
                                    schema_json=json.dumps(schema),
                                    user_intent=example["user_intent"],
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ]
                new_example = {
                    "messages": messages,
                    "target_schema": schema,
                    "target_content": json.dumps(
                        example["structured_output"], ensure_ascii=False
                    ),
                }
                new_example["evaluation_labels"] = example["evaluation_labels"]
                example_id = generate_id(new_example)
                new_example["id"] = example_id
                f.write(json.dumps(new_example, ensure_ascii=False) + "\n")
            except Exception:
                print(f"Failed to convert the data with the image_url {image_url}")


if __name__ == "__main__":
    main()
