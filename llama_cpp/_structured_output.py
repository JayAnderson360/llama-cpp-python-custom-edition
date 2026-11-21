"""Shared structured-output preparation and validation, independent of a model.

Schema-bearing requests never fall back to unconstrained JSON. Strict mode uses
a conservative subset of the bundled GBNF converter; completed responses are
also validated against the full supplied schema. Streaming content is provisional
until the terminal chunk has passed validation.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

from .llama_grammar import JSON_GBNF, LlamaGrammar


class StructuredOutputRequestError(ValueError):
    """An invalid or unsupported structured-output request (HTTP 400)."""

    code = "invalid_response_format"


class StructuredOutputValidationError(ValueError):
    """Generated structured output was incomplete or invalid (HTTP 422)."""

    code = "structured_output_validation_failed"


_ANNOTATIONS = {
    "$schema", "$comment", "title", "description", "default", "examples",
    "deprecated", "readOnly", "writeOnly", "$defs", "definitions",
}
_STRICT_KEYWORDS = _ANNOTATIONS | {
    "type", "$ref", "const", "enum", "anyOf", "properties", "required",
    "additionalProperties", "items", "prefixItems", "minItems", "maxItems",
    "pattern", "format", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum",
}
_SCHEMA_MAPS = {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
_SCHEMA_LISTS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SCHEMA_VALUES = {
    "additionalProperties", "additionalItems", "unevaluatedProperties", "unevaluatedItems",
    "contains", "propertyNames", "not", "if", "then", "else",
}


def _schema_nodes(schema: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    """Visit schema positions, not arbitrary JSON values in const/default/enum."""
    yield path, schema
    if not isinstance(schema, dict):
        return
    for key, value in schema.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            for name, child in value.items():
                yield from _schema_nodes(child, f"{path}.{key}[{name!r}]")
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            for index, child in enumerate(value):
                yield from _schema_nodes(child, f"{path}.{key}[{index}]")
        elif key == "items":
            if isinstance(value, list):
                for index, child in enumerate(value):
                    yield from _schema_nodes(child, f"{path}.items[{index}]")
            else:
                yield from _schema_nodes(value, f"{path}.items")
        elif key in _SCHEMA_VALUES and not (key == "additionalProperties" and isinstance(value, bool)):
            yield from _schema_nodes(value, f"{path}.{key}")
        elif key == "dependencies" and isinstance(value, dict):
            for name, child in value.items():
                if not isinstance(child, list):
                    yield from _schema_nodes(child, f"{path}.dependencies[{name!r}]")


def _check_references(schema: Dict[str, Any]) -> None:
    for path, node in _schema_nodes(schema):
        if not isinstance(node, dict):
            continue
        # Embedded resources and dynamic references do not match this converter's
        # local JSON-pointer resolution model. Do not let a validator fetch them.
        if any(key in node for key in ("$id", "$anchor", "$dynamicAnchor", "$recursiveAnchor", "$dynamicRef", "$recursiveRef")):
            raise StructuredOutputRequestError(f"Unsupported schema reference scope at {path}.")
        if "$ref" not in node:
            continue
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise StructuredOutputRequestError(f"Only local JSON-pointer references are supported at {path}.")
        target: Any = schema
        try:
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                if isinstance(target, list):
                    if not part.isdecimal() or str(int(part)) != part:
                        raise ValueError("Invalid reference array index.")
                    target = target[int(part)]
                else:
                    target = target[part]
        except (KeyError, IndexError, ValueError, TypeError):
            raise StructuredOutputRequestError(f"Unresolved schema reference at {path}.") from None
        if not isinstance(target, (dict, bool)):
            raise StructuredOutputRequestError(f"Reference does not identify a schema at {path}.")


def _check_strict_subset(schema: Dict[str, Any]) -> None:
    """Reject constraints the bundled converter would ignore or approximate."""
    for path, node in _schema_nodes(schema):
        if not isinstance(node, dict):
            raise StructuredOutputRequestError(f"Strict grammar requires an object schema at {path}.")
        unsupported = set(node) - _STRICT_KEYWORDS
        if unsupported:
            raise StructuredOutputRequestError(
                f"Unsupported strict schema keyword {sorted(unsupported)[0]!r} at {path}."
            )
        assertions = set(node) - _ANNOTATIONS
        for exclusive in ("$ref", "anyOf", "enum", "const"):
            if exclusive in node:
                allowed = {exclusive, "type"} if exclusive in ("enum", "const") else {exclusive}
                if assertions - allowed:
                    raise StructuredOutputRequestError(
                        f"Combined constraints with {exclusive!r} are unsupported in strict mode at {path}."
                    )
        kind = node.get("type")
        if isinstance(kind, list) and assertions - {"type"}:
            raise StructuredOutputRequestError(f"Use anyOf for constrained type unions at {path}.")
        if "required" in node and set(node["required"]) - set(node.get("properties", {})):
            raise StructuredOutputRequestError(f"Strict required fields must be declared in properties at {path}.")
        if "prefixItems" in node and any(key in node for key in ("items", "minItems", "maxItems")):
            raise StructuredOutputRequestError(f"Combined tuple constraints are unsupported in strict mode at {path}.")
        if any(key in node for key in ("minItems", "maxItems")) and "items" not in node:
            raise StructuredOutputRequestError(f"Strict array bounds require items at {path}.")
        if isinstance(node.get("items"), list):
            raise StructuredOutputRequestError(f"Use prefixItems for tuples in strict mode at {path}.")
        string_groups = sum(("pattern" in node, "format" in node,
                             "minLength" in node or "maxLength" in node))
        if string_groups > 1:
            raise StructuredOutputRequestError(f"Combined string constraints are unsupported in strict mode at {path}.")
        if ("minLength" in node or "maxLength" in node) and kind != "string":
            raise StructuredOutputRequestError(f"Strict string lengths require type=string at {path}.")
        if "format" in node and node["format"] not in {"uuid", "date", "time", "date-time"}:
            raise StructuredOutputRequestError(f"Unsupported strict string format at {path}.")
        bounds = {key for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum") if key in node}
        if bounds and (kind != "integer" or any(type(node[key]) is not int for key in bounds)):
            raise StructuredOutputRequestError(f"Strict numeric bounds currently require integer values and type at {path}.")
        if {"minimum", "exclusiveMinimum"} <= bounds or {"maximum", "exclusiveMaximum"} <= bounds:
            raise StructuredOutputRequestError(f"Combined inclusive/exclusive bounds are unsupported at {path}.")


def _make_validator(schema: Dict[str, Any], strict: bool) -> Any:
    from jsonschema import Draft202012Validator, FormatChecker
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for
    from referencing import Registry
    from referencing.exceptions import NoSuchResource

    if "$schema" in schema and validator_for(schema, default=None) is None:
        raise StructuredOutputRequestError("Unsupported JSON Schema dialect.")
    validator_class = validator_for(schema, default=Draft202012Validator)
    try:
        validator_class.check_schema(schema)
    except SchemaError as exc:
        raise StructuredOutputRequestError("Invalid JSON Schema definition.") from exc
    _check_references(schema)
    if strict:
        _check_strict_subset(schema)
        for path, node in _schema_nodes(schema):
            if isinstance(node, dict) and ("enum" in node or "const" in node):
                values = node.get("enum", [node.get("const")])
                if any(not validator_class(node).is_valid(value) for value in values):
                    raise StructuredOutputRequestError(f"Inconsistent strict enum/const constraints at {path}.")

    def no_external_resources(uri: str) -> Any:
        raise NoSuchResource(ref=uri)

    checker = FormatChecker()
    if strict:
        for path, node in _schema_nodes(schema):
            if isinstance(node, dict) and "format" in node and node["format"] not in checker.checkers:
                raise StructuredOutputRequestError(f"Strict format validator is unavailable at {path}.")
    return validator_class(schema, format_checker=checker, registry=Registry(retrieve=no_external_resources))


def _reject_constant(value: str) -> Any:
    raise ValueError("Non-finite JSON number.")


def _unique_object(pairs: Any) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key.")
        value[key] = item
    return value


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON number.")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


@dataclass(frozen=True)
class PreparedStructuredOutput:
    response_format: Dict[str, Any]
    grammar: LlamaGrammar
    validator: Any = None

    def validate_content(self, content: Any, finish_reason: Any = None) -> None:
        if finish_reason is not None and finish_reason != "stop":
            raise StructuredOutputValidationError("Structured output did not finish as a complete JSON response.")
        if not isinstance(content, str):
            raise StructuredOutputValidationError("Structured output requires textual JSON content.")
        try:
            value = json.loads(content, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
            _check_finite(value)
        except (ValueError, TypeError, RecursionError) as exc:
            raise StructuredOutputValidationError("Generated content is not complete, unambiguous JSON.") from exc
        if self.validator is not None:
            from jsonschema.exceptions import ValidationError

            try:
                self.validator.validate(value)
            except ValidationError as exc:
                path = "$" + "".join(f"[{json.dumps(part)}]" for part in exc.absolute_path)
                raise StructuredOutputValidationError(
                    f"Generated JSON violates {exc.validator!r} at {path}."
                ) from exc
            except Exception as exc:
                raise StructuredOutputValidationError("Could not validate generated JSON against the schema.") from exc


def prepare_response_format(response_format: Any) -> Optional[PreparedStructuredOutput]:
    if response_format is None:
        return None
    if not isinstance(response_format, Mapping):
        raise StructuredOutputRequestError("response_format must be an object.")
    kind = response_format.get("type")
    if kind == "text":
        if "schema" in response_format or "json_schema" in response_format:
            raise StructuredOutputRequestError("Text response format cannot include a JSON schema.")
        return None
    if kind not in ("json_object", "json_schema"):
        raise StructuredOutputRequestError("Unsupported response_format type.")
    definition: Mapping[str, Any] = response_format
    if kind == "json_schema":
        definition = response_format.get("json_schema")
        if not isinstance(definition, Mapping) or "schema" not in definition:
            raise StructuredOutputRequestError("json_schema.schema is required.")
        if "schema" in response_format and response_format["schema"] != definition["schema"]:
            raise StructuredOutputRequestError("Conflicting response schemas.")
        if "strict" in response_format and response_format["strict"] != definition.get("strict", False):
            raise StructuredOutputRequestError("Conflicting strict settings.")
    elif "json_schema" in response_format:
        raise StructuredOutputRequestError("Use type=json_schema for a nested schema.")
    strict = definition.get("strict", False)
    if strict is None:
        strict = False
    if type(strict) is not bool:
        raise StructuredOutputRequestError("strict must be a boolean.")
    for key in ("name", "description"):
        if key in definition and not isinstance(definition[key], str):
            raise StructuredOutputRequestError(f"json_schema.{key} must be a string.")
    if "schema" not in definition:
        if strict:
            raise StructuredOutputRequestError("Strict structured output requires a schema.")
        return PreparedStructuredOutput({"type": "json_object"}, LlamaGrammar.from_string(JSON_GBNF, verbose=False))
    schema = definition["schema"]
    if not isinstance(schema, dict):
        raise StructuredOutputRequestError("The response schema must be a JSON object.")
    try:
        schema = json.loads(json.dumps(schema, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise StructuredOutputRequestError("The response schema must contain finite JSON data.") from exc
    validator = _make_validator(schema, strict)
    try:
        grammar = LlamaGrammar.from_json_schema(copy.deepcopy(schema), verbose=False)
    except Exception as exc:
        raise StructuredOutputRequestError("The response schema cannot be converted to a generation grammar.") from exc
    canonical = {"type": "json_schema", "json_schema": {
        "name": definition.get("name", "response"), "schema": schema, "strict": strict,
    }}
    return PreparedStructuredOutput(canonical, grammar, validator)


def check_structured_output_options(prepared: Any, *, tools: Any = None,
                                    tool_choice: Any = None, functions: Any = None,
                                    function_call: Any = None, grammar: Any = None) -> None:
    if prepared is None:
        return
    if grammar is not None:
        raise StructuredOutputRequestError("Choose response_format or an explicit grammar, not both.")
    if (tools and tool_choice != "none") or (functions and function_call != "none"):
        raise StructuredOutputRequestError("Schema responses cannot be combined with active tool/function calling.")


def validate_chat_result(result: Any, prepared: Optional[PreparedStructuredOutput], *, stream: bool) -> Any:
    if prepared is None:
        return result
    if stream:
        return _validated_stream(result, prepared)
    if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
        raise StructuredOutputValidationError("Structured completion has no choices.")
    for choice in result["choices"]:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise StructuredOutputValidationError("Structured completion has no assistant message.")
        # A refusal is an explicit protocol outcome, not a schema-valid answer.
        if message.get("refusal"):
            continue
        if message.get("tool_calls") or message.get("function_call"):
            raise StructuredOutputValidationError("A tool call is not a schema-valid JSON answer.")
        prepared.validate_content(message.get("content"), choice.get("finish_reason"))
    return result


def _validated_stream(chunks: Any, prepared: PreparedStructuredOutput) -> Iterator[Any]:
    states: Dict[int, Dict[str, Any]] = {}
    try:
        for chunk in chunks:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
                raise StructuredOutputValidationError("Malformed structured-output stream chunk.")
            for choice in chunk["choices"]:
                if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] < 0:
                    raise StructuredOutputValidationError("Stream choices require integer indexes.")
                index = choice["index"]
                state = states.setdefault(index, {"parts": [], "refusal": False, "finished": False})
                if state["finished"]:
                    raise StructuredOutputValidationError("Stream choice continued after its terminal chunk.")
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise StructuredOutputValidationError("Malformed structured-output stream delta.")
                if delta.get("tool_calls") or delta.get("function_call"):
                    raise StructuredOutputValidationError("A tool call is not a structured JSON stream.")
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise StructuredOutputValidationError("Stream content must be text.")
                    state["parts"].append(content)
                state["refusal"] = state["refusal"] or bool(delta.get("refusal"))
                reason = choice.get("finish_reason")
                if reason is not None:
                    if not state["refusal"]:
                        prepared.validate_content("".join(state["parts"]), reason)
                    state["finished"] = True
                    state["parts"] = []
            yield chunk
        if not states or any(not state["finished"] for state in states.values()):
            raise StructuredOutputValidationError("Structured-output stream ended before completion.")
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()
