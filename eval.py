#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Relaxed evaluation for structured output generation with evaluation labels.

This module provides relaxed evaluation capabilities for structured output,
supporting different evaluation strategies based on evaluation labels:

- "exact": Same as original function (exact matching)
- "fuzzy": Use normalized edit distance for strings (>0.8 = match),
           relative error for numbers (<0.05 = match)
- "ignore": Skip these fields in evaluation
"""

import json
from typing import Any, Dict, List, Union
from jsonschema import validate
import re

# Choosing 0.8 and 0.05 as a threshold due to common usage
# Reference: ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering
# https://arxiv.org/abs/2504.05506
STRING_SIMILARITY = 0.8
RELATIVE_ERROR = 0.05


def _extract_from_markdown_code_block(text: str) -> str:
    """Extract JSON content from markdown code blocks.

    Handles various markdown formats:
    - ```json ... ```
    - ```format ... ```
    - ``` ... ```

    The ending can be either \n``` or just ```

    Args:
        text: The text that may contain markdown code blocks

    Returns:
        The extracted content if found in markdown blocks, otherwise the original text
    """
    pattern = r"```(?:[a-zA-Z]*\n)?(.*?)(?:\n)?```"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _normalize_string(s: str) -> str:
    """Normalize string for comparison according to BFCL specifications.
        - Remove whitespace
        - Strip punctuation ,./-_*^
        - Convert to lowercase

    Args:
        s: Raw string input.

    Returns:
        Normalized string.
    """
    if not isinstance(s, str):
        return s
    # Remove whitespace
    s = re.sub(r"\s+", "", s)
    # Strip specified punctuation (including common ones like ! ? : ; etc.)
    s = re.sub(r'[,./_*^!?:;()[\]{}"\'\\-]', "", s)
    # Convert to lowercase
    return s.lower()


def _sort_structured_output(obj: dict[str, Any]) -> dict[str, Any]:
    """Recursively normalize a JSON object for consistent comparison.

    This function ensures that dictionary keys are sorted consistently
    for reliable comparison between JSON objects.

    Args:
        obj: The JSON object to normalize (dict, list, or primitive)

    Returns:
        The normalized object with consistent key ordering
    """
    if isinstance(obj, dict):
        # Sort dictionary by keys and recursively normalize values
        return {k: _sort_structured_output(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # Recursively normalize each item in the list
        return [_sort_structured_output(item) for item in obj]
    else:
        # Return primitive values as-is
        return obj


def _compute_field_match(
    pred: dict[str, Any], ground_truth: dict[str, Any]
) -> dict[str, Any]:
    """Compute field match score for nested structured output.

    This function uses Abstract Syntax Tree (AST) substring matching method, by checking each fields between pred and ground truth recursively.
    Reference: The Berkeley Function Calling Leaderboard (BFCL) https://openreview.net/pdf?id=2GmDdhBdDk

        Matching criteria:

        STRING
        - Comparison is case-insensitive.
        - All strings are standardized before checking:
            – whitespace removed,
            – punctuation ,./-_*ˆ (note: and ˆ) stripped

        DICTIONARY (dict)
        - Key presence and value correctness are checked.
        - Key order is ignored (dictionaries are inherently unordered).

        LIST
        - Order matters: [1,2,3] != [2,3,1]
        - Type matching is recursive for nested structures; outer and inner element types must satisfy the specification.

        Numbers (int or float)
        - All converted to float and check exact match.

    Args:
        pred: Predicted structured output.
        ground_truth: Ground truth structured output.

    Returns:
        Field match metrics score.
    """

    def _compare_values(pred_val: Any, gt_val: Any, results: dict) -> bool:  # noqa: PLR0911
        """
        Recursively compare two values and update results.
        Returns True if values match according to the criteria.
        """
        results["total_num_fields"] += 1

        # Handle None values
        if pred_val is None and gt_val is None:
            # Both are None - this is a match, count it as a string match
            results["num_field_match_string"] += 1
            return True
        if pred_val is None or gt_val is None:
            return False

        # string comparison
        if isinstance(pred_val, str) and isinstance(gt_val, str):
            normalized_pred = _normalize_string(pred_val)
            normalized_gt = _normalize_string(gt_val)
            match = normalized_pred == normalized_gt
            if match:
                results["num_field_match_string"] += 1
            return match

        # number comparison (int or float)
        if isinstance(pred_val, (int, float)) and isinstance(gt_val, (int, float)):
            try:
                pred_float = float(pred_val)
                gt_float = float(gt_val)
                match = pred_float == gt_float
                if match:
                    results["num_field_match_numbers"] += 1
                return match
            except (ValueError, TypeError):
                return False

        # list comparison
        if isinstance(pred_val, list) and isinstance(gt_val, list):
            if len(pred_val) != len(gt_val):
                return False

            all_match = True
            for pred_item, gt_item in zip(pred_val, gt_val):
                if not _compare_values(pred_item, gt_item, results):
                    all_match = False

            if all_match:
                results["num_field_match_list"] += 1
            return all_match

        # dict comparison
        if isinstance(pred_val, dict) and isinstance(gt_val, dict):
            gt_keys = set(gt_val.keys())
            pred_keys = set(pred_val.keys())

            # Count individual field matches within the dictionary
            dict_matches = 0
            total_gt_fields = len(gt_keys)

            for key in gt_keys:
                if key in pred_keys:
                    if _compare_values(pred_val[key], gt_val[key], results):
                        dict_matches += 1
            # A dictionary matches if all its fields match
            if dict_matches == total_gt_fields and total_gt_fields > 0:
                results["num_field_match_dict"] += 1
                return True

            return dict_matches > 0  # Return true if at least some fields matched

        # Type mismatch or unsupported types
        return pred_val == gt_val

    # Initialize results
    metrics = {
        "total_num_fields": 0,
        "num_field_match_list": 0,
        "num_field_match_dict": 0,
        "num_field_match_numbers": 0,
        "num_field_match_string": 0,
    }

    _compare_values(pred, ground_truth, metrics)

    metrics["num_field_match"] = (
        metrics["num_field_match_list"]
        + metrics["num_field_match_dict"]
        + metrics["num_field_match_numbers"]
        + metrics["num_field_match_string"]
    )

    return metrics


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Calculate normalized edit distance on word level (1 - distance/max_length)."""
    if not s1 and not s2:
        return 1.0

    # Tokenize strings by spaces to work on word level
    words1 = s1.split()
    words2 = s2.split()

    max_len = max(len(words1), len(words2))
    if max_len == 0:
        return 1.0

    # Calculate edit distance between word lists
    distance = _word_level_levenshtein(words1, words2)
    return 1.0 - (distance / max_len)


def _word_level_levenshtein(words1: list, words2: list) -> int:
    """Calculate Levenshtein distance between two lists of words."""
    if len(words1) < len(words2):
        # Swap arguments to ensure words1 is the longer list
        return _word_level_levenshtein(words2, words1)

    if len(words2) == 0:
        return len(words1)

    previous_row = list(range(len(words2) + 1))
    for i, w1 in enumerate(words1):
        current_row = [i + 1]
        for j, w2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _relative_error(pred_val: float, gt_val: float) -> float:
    """Calculate relative error between two numbers."""
    if pred_val == gt_val:
        return 0.0
    denominator = max(abs(pred_val), abs(gt_val))
    return abs(pred_val - gt_val) / denominator


def _compute_field_match_with_labels(
    pred: dict[str, Any],
    ground_truth: dict[str, Any],
    evaluation_labels: dict[str, Any],
) -> dict[str, Any]:
    """Compute field match score for nested structured output using evaluation labels.

    This function extends the original _compute_field_match by supporting different
    evaluation strategies based on evaluation_labels:
    - "exact": Same as original function (exact matching)
    - "fuzzy": Use normalized edit distance for strings (>0.8 = match),
               relative error for numbers (<0.05 = match)
    - "ignore": Skip these fields in evaluation

    Args:
        pred: Predicted structured output.
        ground_truth: Ground truth structured output.
        evaluation_labels: Labels defining evaluation strategy for each field.

    Returns:
        Field match metrics score with fuzzy matching support.
    """

    def _is_meaningful_label(label: Any) -> bool:
        """
        Check if an evaluation label indicates a meaningful comparison.
        Returns False if the label indicates the field should be ignored.
        Handles nested structures recursively.
        """
        if isinstance(label, str):
            return label != "ignore"
        elif isinstance(label, dict):
            # For dict labels, meaningful if any value is meaningful (recursive check)
            return any(_is_meaningful_label(v) for v in label.values())
        elif isinstance(label, list):
            # For list labels, meaningful if any item is meaningful (recursive check)
            return any(_is_meaningful_label(item) for item in label)
        else:
            # Unknown label type, assume meaningful
            return True

    def _compare_values_with_labels(  # noqa: PLR0911
        pred_val: Any, gt_val: Any, eval_label: Union[str, dict, list], results: dict
    ) -> tuple[bool, bool]:
        """
        Compare two values using the specified evaluation strategy.
        Returns (match_result, is_meaningful) where:
        - match_result: True if values match according to the criteria
        - is_meaningful: True if this comparison was meaningful (not ignored)
        """
        # Skip ignored fields
        if eval_label == "ignore":
            return True, False  # Match but not meaningful

        results["total_num_fields"] += 1

        # Handle None values
        if pred_val is None and gt_val is None:
            # Both are None - this is a match, but we need to count it
            # We'll count it as a string match for consistency
            results["num_field_match_string"] += 1
            return True, True
        if pred_val is None or gt_val is None:
            return False, True

        # String comparison
        if isinstance(pred_val, str) and isinstance(gt_val, str):
            if eval_label == "exact":
                normalized_pred = _normalize_string(pred_val)
                normalized_gt = _normalize_string(gt_val)
                match = normalized_pred == normalized_gt
            elif eval_label == "fuzzy":
                # Use normalized edit distance for fuzzy string matching
                similarity = _normalized_edit_distance(pred_val, gt_val)
                match = similarity > STRING_SIMILARITY
            else:
                # Default to exact matching for unknown labels
                normalized_pred = _normalize_string(pred_val)
                normalized_gt = _normalize_string(gt_val)
                match = normalized_pred == normalized_gt

            if match:
                results["num_field_match_string"] += 1
            return match, True

        # Number comparison (int or float)
        if isinstance(pred_val, (int, float)) and isinstance(gt_val, (int, float)):
            try:
                pred_float = float(pred_val)
                gt_float = float(gt_val)

                if eval_label == "exact":
                    match = pred_float == gt_float
                elif eval_label == "fuzzy":
                    # Use relative error for fuzzy number matching
                    rel_error = _relative_error(pred_float, gt_float)
                    match = rel_error < RELATIVE_ERROR
                else:
                    # Default to exact matching
                    match = pred_float == gt_float

                if match:
                    results["num_field_match_numbers"] += 1
                return match, True
            except (ValueError, TypeError):
                return False, True

        # List comparison
        if isinstance(pred_val, list) and isinstance(gt_val, list):
            # First check if the entire list has meaningful labels
            if not _is_meaningful_label(eval_label):
                results["total_num_fields"] -= 1  # Remove the list field count
                return False, False

            # If meaningful, proceed with the cleaner logic from _prev version
            if len(pred_val) != len(gt_val):
                return False, True

            # For lists, eval_label should also be a list
            has_meaningful_items = False
            if isinstance(eval_label, list) and len(eval_label) == len(gt_val):
                all_match = True
                for pred_item, gt_item, item_label in zip(pred_val, gt_val, eval_label):
                    match, meaningful = _compare_values_with_labels(
                        pred_item, gt_item, item_label, results
                    )
                    if meaningful:
                        has_meaningful_items = True
                    if not match:
                        all_match = False
            else:
                # If eval_label is not a list or doesn't match length, use it for all items
                all_match = True
                for pred_item, gt_item in zip(pred_val, gt_val):
                    match, meaningful = _compare_values_with_labels(
                        pred_item, gt_item, eval_label, results
                    )
                    if meaningful:
                        has_meaningful_items = True
                    if not match:
                        all_match = False

            # Only count as meaningful if at least one item was meaningful
            if not has_meaningful_items:
                results["total_num_fields"] -= 1  # Remove the list field count
                return False, False

            if all_match:
                results["num_field_match_list"] += 1
            return all_match, True

        # Dict comparison
        if isinstance(pred_val, dict) and isinstance(gt_val, dict):
            gt_keys = set(gt_val.keys())
            pred_keys = set(pred_val.keys())

            dict_matches = 0
            meaningful_fields = 0

            # Process ALL keys in ground truth to ensure complete evaluation
            for key in gt_keys:
                # Get evaluation label for this key
                key_eval_label = (
                    eval_label.get(key, "exact")
                    if isinstance(eval_label, dict)
                    else eval_label
                )

                # Check if this key is meaningful using the improved function
                if _is_meaningful_label(key_eval_label):
                    meaningful_fields += 1

                    if key in pred_keys:
                        # Key exists in both - compare values recursively
                        match, _ = _compare_values_with_labels(
                            pred_val[key], gt_val[key], key_eval_label, results
                        )
                        if match:
                            dict_matches += 1
                    # If key is missing from prediction, it's automatically a non-match
                    # (dict_matches is not incremented)

            # If no fields were meaningful (all ignored), don't count this dict
            if meaningful_fields == 0:
                results["total_num_fields"] -= 1  # Remove the dict field count
                return False, False

            # A dictionary matches if ALL its meaningful fields match
            if dict_matches == meaningful_fields:
                results["num_field_match_dict"] += 1
                return True, True

            return dict_matches > 0, True

        # Type mismatch or unsupported types - fall back to exact comparison
        return pred_val == gt_val, True

    # Initialize results
    metrics = {
        "total_num_fields": 0,
        "num_field_match_list": 0,
        "num_field_match_dict": 0,
        "num_field_match_numbers": 0,
        "num_field_match_string": 0,
    }

    _, _ = _compare_values_with_labels(pred, ground_truth, evaluation_labels, metrics)

    metrics["num_field_match"] = (
        metrics["num_field_match_list"]
        + metrics["num_field_match_dict"]
        + metrics["num_field_match_numbers"]
        + metrics["num_field_match_string"]
    )

    return metrics


def metric_fn(
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Metric function for relaxed structured output evaluation.

    This function evaluates structured output using relaxed matching criteria
    when evaluation_labels are provided in the response data.

    Args:
        responses: List of response dictionaries containing predictions and targets

    Returns:
        Dictionary containing evaluation metrics
    """
    exact_match = []
    followed_schema = []
    num_invalid_generation = 0
    num_fields = 0
    num_field_match = 0
    full_match = []

    for response in responses:
        if "generation" not in response:
            num_invalid_generation += 1
            exact_match.append(False)
            followed_schema.append(False)
            continue
        pred = response["generation"].strip()

        # Clean up common model output artifacts.
        pred = pred.strip()
        pred = pred.rstrip("<turn_end>")

        # Extract JSON from markdown code blocks if present
        pred = _extract_from_markdown_code_block(pred)

        ref = json.loads(response["target_content"])
        if (
            "response_format" in response
            and "json_schema" in response["response_format"]
            and "schema" in response["response_format"]["json_schema"]
        ):
            schema = response["response_format"]["json_schema"]["schema"]
        elif "target_schema" in response:
            schema = response["target_schema"]
        else:
            raise ValueError(f"Cannot find json schema in response: {response}")
        try:
            pred_d = json.loads(pred)
        except json.JSONDecodeError:
            num_invalid_generation += 1
            exact_match.append(False)
            followed_schema.append(False)
            continue
        # Normalize both prediction and reference for consistent comparison
        normalized_pred = _sort_structured_output(pred_d)
        normalized_ref = _sort_structured_output(ref)

        is_exact_match = normalized_pred == normalized_ref
        exact_match.append(is_exact_match)

        # Use relaxed field matching if evaluation_labels are provided
        if "evaluation_labels" in response:
            field_match = _compute_field_match_with_labels(
                pred_d, ref, response["evaluation_labels"]
            )
        else:
            print(
                "No evaluation labels found in the response. Falling back to exact field matching."
            )
            # Fall back to exact field matching if no evaluation labels
            field_match = _compute_field_match(pred_d, ref)

        num_fields += field_match["total_num_fields"]
        num_field_match += field_match["num_field_match"]
        full_match.append(
            field_match["total_num_fields"] == field_match["num_field_match"]
        )

        # schema verification
        try:
            validate(instance=pred_d, schema=schema)
            followed_schema.append(True)
        except Exception:
            followed_schema.append(False)

    output_metrics = {
        "num_examples": len(responses),
        "num_invalid_generation": num_invalid_generation,
        "exact_match": sum(exact_match) * 1.0 / max(1, len(exact_match)),
        "followed_schema": sum(followed_schema) * 1.0 / max(1, len(followed_schema)),
        "relaxed_field_match_accuracy": float(num_field_match)
        / max(1, float(num_fields)),
        "relaxed_all_fields_match_accuracy": sum(full_match)
        * 1.0
        / max(1, len(full_match)),
    }

    return output_metrics
