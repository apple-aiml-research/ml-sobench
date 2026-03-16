#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Core components to load SO-Bench structured output dataset"""

from __future__ import annotations  # noqa: I001

import io
import json
import logging
import os
from typing import Any, Optional, Union

from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from PIL import Image
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class StructuredOutput(BaseModel):
    """Core data structure to store structured output.

    This class serves as the central structure for organizing structured output data,
    """

    model_config = ConfigDict(extra="ignore")

    image_url: str

    schema_json: dict[str, Any]

    structured_output: Union[dict[str, Any], list[dict[str, Any]]]

    user_intent: str

    # evaluation labels with same structure as structured_output but primitive values replaced with "exact", "fuzzy", "ignore"
    evaluation_labels: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None

    def model_post_init(self, __context: Any) -> None:  # pylint: disable=arguments-differ
        """Post-initialization validation (Pydantic v2 method)."""
        # Validate json schemas
        try:
            Validator = validator_for(self.schema_json)
            Validator.check_schema(self.schema_json)
        except Exception as e:
            raise SchemaError(
                f"Input schema is not a valid json schema: {str(e)}"
            ) from e

        # Validate structure outputs
        try:
            jsonschema_validate(
                instance=self.structured_output, schema=self.schema_json
            )
        except Exception as e:
            raise SchemaError(
                f"Structured output does not follow the schema: {str(e)}"
            ) from e

        # Validate evaluation_labels if present
        if self.evaluation_labels is not None:
            self._validate_evaluation_labels()

    def _validate_evaluation_labels(self) -> None:
        """Validate that evaluation_labels follow the schema structure and contain valid values."""
        if self.schema_json is None:
            raise ValueError(
                "schema_json must be provided when evaluation_labels is specified"
            )

        if self.structured_output is None:
            raise ValueError(
                "structured_output must be provided when evaluation_labels is specified"
            )

        # Convert None values to "ignore" and validate structure
        try:
            self.evaluation_labels = self._normalize_and_validate_evaluation_structure(
                self.evaluation_labels, self.structured_output, "evaluation_labels"
            )
        except Exception as e:
            raise ValueError(f"Invalid evaluation_labels structure: {str(e)}") from e

    def _normalize_and_validate_evaluation_structure(
        self, eval_labels: Any, structured_data: Any, path: str
    ) -> Any:
        """Recursively normalize None values to 'ignore' and validate evaluation labels structure and values."""
        valid_eval_values = {"exact", "fuzzy", "ignore"}

        if isinstance(structured_data, dict) and isinstance(eval_labels, dict):
            # Both should be dictionaries with matching keys
            if set(eval_labels.keys()) != set(structured_data.keys()):
                raise ValueError(
                    f"Keys mismatch at {path}: evaluation_labels keys {set(eval_labels.keys())} "
                    f"do not match structured_output keys {set(structured_data.keys())}"
                )

            normalized_dict = {}
            for key in structured_data.keys():
                normalized_dict[key] = (
                    self._normalize_and_validate_evaluation_structure(
                        eval_labels[key], structured_data[key], f"{path}.{key}"
                    )
                )
            return normalized_dict

        elif isinstance(structured_data, list) and isinstance(eval_labels, list):
            # Both should be lists with same length
            if len(eval_labels) != len(structured_data):
                raise ValueError(
                    f"Length mismatch at {path}: evaluation_labels has {len(eval_labels)} items "
                    f"but structured_output has {len(structured_data)} items"
                )

            normalized_list = []
            for i, (eval_item, struct_item) in enumerate(
                zip(eval_labels, structured_data)
            ):
                normalized_list.append(
                    self._normalize_and_validate_evaluation_structure(
                        eval_item, struct_item, f"{path}[{i}]"
                    )
                )
            return normalized_list

        else:
            # Leaf node - handle None values by converting to "ignore"
            if eval_labels is None:
                return "ignore"

            # Validate that eval_labels is a valid evaluation value
            if not isinstance(eval_labels, str) or eval_labels not in valid_eval_values:
                raise ValueError(
                    f"Invalid evaluation label at {path}: '{eval_labels}'. "
                    f"Must be one of {valid_eval_values}"
                )
            return eval_labels

    @property
    def image(self) -> Image.Image:
        """Get the image from the image_url."""
        with open(self.image_url, "rb") as fd:
            return Image.open(io.BytesIO(fd.read()))

    @classmethod
    def from_file(cls, filename: str) -> "StructuredOutput":
        """Load StructuredOutput from a JSON file.

        Args:
            filename: Path to the JSON file (supports both local and remote paths)

        Returns:
            StructuredOutput instance
        """
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            raise ValueError(
                f"Failed to load StructuredOutput from {filename}: {str(e)}"
            ) from e

    def to_file(self, filename: str) -> None:
        """Save StructuredOutput to a JSON file.

        Args:
            filename: Path where to save the JSON file (supports both local and remote paths)

        Raises:
            OSError: If file cannot be written
        """
        try:
            data = self.model_dump()
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise OSError(
                f"Failed to save StructuredOutput to {filename}: {str(e)}"
            ) from e
