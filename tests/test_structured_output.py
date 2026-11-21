"""Model-free regression tests; run on the designated testing machine.

Load the actual pure-Python implementation in an isolated package so these tests
do not need a compiled llama library, a model, or a GPU. HTTP tests execute the
real chat route, publisher, request types, and error handler with a fake model.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import gc
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
import warnings
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "_structured_output_unit"
PACKAGE = types.ModuleType(PREFIX)
PACKAGE.__path__ = [str(ROOT / "llama_cpp")]
sys.modules[PREFIX] = PACKAGE


def load_module(relative):
    path = ROOT / "llama_cpp" / (relative.replace(".", "/") + ".py")
    name = PREFIX + "." + relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "llama_cpp":
                    alias.name, alias.asname = PREFIX, "llama_cpp"
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("llama_cpp."):
            node.module = PREFIX + node.module[len("llama_cpp"):]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


grammar = load_module("llama_grammar")
structured = load_module("_structured_output")
llama_types = load_module("llama_types")
for _name, _value in vars(llama_types).items():
    if not _name.startswith("_"):
        setattr(PACKAGE, _name, _value)
PACKAGE.LlamaGrammar = grammar.LlamaGrammar
server_package = types.ModuleType(PREFIX + ".server")
server_package.__path__ = [str(ROOT / "llama_cpp" / "server")]
sys.modules[server_package.__name__] = server_package
server_types = load_module("server.types")
server_errors = load_module("server.errors")


def extract_functions(path, names, namespace, method=False):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body
    if method:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == "Llama").body
    selected = [copy.deepcopy(node) for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    for node in selected:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


METHOD_NAMESPACE = dict(vars(structured), __name__=PREFIX + ".model_test")
extract_functions(ROOT / "llama_cpp" / "llama.py", {"create_chat_completion"}, METHOD_NAMESPACE, method=True)


class FakeLlama:
    create_chat_completion = METHOD_NAMESPACE["create_chat_completion"]

    def __init__(self, handler):
        self.chat_handler = handler
        self.chat_format = "test"
        self._chat_handlers = {}


PACKAGE.Llama = FakeLlama


def envelope(content, reason="stop", **message):
    return {"id": "test", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, **message},
                         "finish_reason": reason}]}


def chunks(content, reason="stop"):
    yield {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
           "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
    yield {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
           "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def request_format(schema, strict=True):
    return {"type": "json_schema", "json_schema": {"name": "dynamic", "schema": schema, "strict": strict}}


PERSON = {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}},
          "required": ["age"], "additionalProperties": False}


def http_app(handler):
    import anyio
    from fastapi import APIRouter, Body, FastAPI, HTTPException, Request, status
    from sse_starlette.sse import EventSourceResponse
    from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
    from functools import partial

    model = FakeLlama(handler)

    async def get_llama_proxy():
        yield lambda model_name: model

    def get_server_settings():
        yield types.SimpleNamespace(interrupt_requests=False)

    namespace = dict(
        __name__=PREFIX + ".http_test", json=json, contextlib=contextlib,
        typing=__import__("typing"), anyio=anyio, llama_cpp=PACKAGE,
        Request=Request, Body=Body, HTTPException=HTTPException, status=status,
        CreateChatCompletionRequest=server_types.CreateChatCompletionRequest,
        CreateCompletionRequest=server_types.CreateCompletionRequest,
        get_llama_proxy=get_llama_proxy, get_server_settings=get_server_settings,
        EventSourceResponse=EventSourceResponse, partial=partial,
        run_in_threadpool=run_in_threadpool, iterate_in_threadpool=iterate_in_threadpool,
        _ping_message_factory=None, llama_outer_lock=anyio.Lock(),
        structured_output_error=server_errors.structured_output_error,
        **{name: getattr(structured, name) for name in (
            "prepare_response_format", "check_structured_output_options",
            "StructuredOutputRequestError", "StructuredOutputValidationError")},
    )
    extract_functions(ROOT / "llama_cpp" / "server" / "app.py",
                      {"create_chat_completion", "prepare_request_resources", "get_event_publisher", "_receive_chat_events"}, namespace)
    router = APIRouter(route_class=server_errors.RouteErrorHandler)
    router.add_api_route("/v1/chat/completions", namespace["create_chat_completion"], methods=["POST"], response_model=None)
    app = FastAPI()
    app.include_router(router)
    return app


class StructuredOutputTests(unittest.TestCase):
    def test_dynamic_schema_and_legacy_envelope(self):
        for schema, value in [(PERSON, {"age": 4}),
                              ({"type": "array", "items": {"type": "string"}}, ["a", "b"])]:
            for response_format in [request_format(schema), {"type": "json_object", "schema": schema, "strict": True}]:
                with self.subTest(schema=schema, response_format=response_format):
                    prepared = structured.prepare_response_format(response_format)
                    prepared.validate_content(json.dumps(value))
                    self.assertEqual(prepared.response_format["json_schema"]["schema"], schema)

    def test_nested_nullable_enum_and_local_reference(self):
        schema = {"$defs": {"Address": {"type": "object", "properties": {"city": {"type": "string"}},
                                       "required": ["city"], "additionalProperties": False}},
                  "type": "object", "properties": {
                      "address": {"$ref": "#/$defs/Address"},
                      "kind": {"enum": ["home", "work"]},
                      "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                  "required": ["address", "kind"], "additionalProperties": False}
        original = copy.deepcopy(schema)
        prepared = structured.prepare_response_format(request_format(schema))
        prepared.validate_content('{"address":{"city":"A"},"kind":"home","note":null}')
        self.assertEqual(schema, original)
        with self.assertRaises(structured.StructuredOutputValidationError):
            prepared.validate_content('{"address":{"city":"A"},"kind":"other"}')

    def test_missing_and_malformed_schema_fail(self):
        for value in [{"type": "json_schema"}, {"type": "json_schema", "json_schema": {}},
                      request_format(None), request_format({"type": "impossible"}),
                      {"type": "json_object", "strict": True}, {"type": "unknown"}]:
            with self.subTest(value=value), self.assertRaises(structured.StructuredOutputRequestError):
                structured.prepare_response_format(value)

    def test_conflicting_schema_or_strictness_fail(self):
        value = request_format(PERSON)
        value["schema"] = {"type": "string"}
        with self.assertRaises(structured.StructuredOutputRequestError):
            structured.prepare_response_format(value)
        value = request_format(PERSON)
        value["strict"] = False
        with self.assertRaises(structured.StructuredOutputRequestError):
            structured.prepare_response_format(value)

    def test_strict_unsupported_constraints_fail_before_generation(self):
        schemas = [{"type": "number", "minimum": 0.5},
                   {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
                   {"oneOf": [{"type": "integer"}, {"type": "number"}]},
                   {"type": "string", "pattern": "^a+$", "minLength": 2},
                   {"type": "integer", "enum": ["wrong type"]},
                   {"type": "object", "required": ["undeclared"]}]
        for schema in schemas:
            with self.subTest(schema=schema), self.assertRaises(structured.StructuredOutputRequestError):
                structured.prepare_response_format(request_format(schema))

    def test_non_strict_still_validates_decimal_bounds(self):
        prepared = structured.prepare_response_format(request_format({"type": "number", "minimum": 0.5}, strict=False))
        prepared.validate_content("0.75")
        with self.assertRaises(structured.StructuredOutputValidationError):
            prepared.validate_content("0.25")

    def test_conversion_failure_never_downgrades(self):
        with patch.object(grammar.LlamaGrammar, "from_json_schema", side_effect=ValueError("unsupported")):
            with self.assertRaises(structured.StructuredOutputRequestError):
                structured.prepare_response_format(request_format(PERSON))

    def test_external_and_unresolved_refs_fail(self):
        for schema in [{"$ref": "https://example.invalid/schema"}, {"$ref": "#/$defs/missing"},
                       {"$id": "https://example.invalid/", "type": "object"}]:
            with self.subTest(schema=schema), self.assertRaises(structured.StructuredOutputRequestError):
                structured.prepare_response_format(request_format(schema))

    def test_const_data_is_not_mistaken_for_a_reference(self):
        prepared = structured.prepare_response_format(request_format({"const": {"$ref": "literal data"}}))
        prepared.validate_content('{"$ref":"literal data"}')

    def test_property_names_and_escaped_reference_pointers(self):
        schema = {"$defs": {"slash/name": {"type": "string"}}, "type": "object",
                  "properties": {"$ref": {"$ref": "#/$defs/slash~1name"}},
                  "required": ["$ref"], "additionalProperties": False}
        prepared = structured.prepare_response_format(request_format(schema))
        prepared.validate_content('{"$ref":"ordinary property"}')

    def test_root_reference_resolves_nested_definitions(self):
        schema = {"$defs": {"Item": {"type": "integer"}, "Answer": {
                      "type": "object", "properties": {"value": {"$ref": "#/$defs/Item"}},
                      "required": ["value"], "additionalProperties": False}},
                  "$ref": "#/$defs/Answer"}
        prepared = structured.prepare_response_format(request_format(schema))
        prepared.validate_content('{"value":4}')

    def test_completed_format_and_array_constraints(self):
        for schema, valid, invalid in [
            ({"type": "string", "format": "date"}, '"2026-11-18"', '"2026-02-31"'),
            ({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}, '[1,2]', '[1,2,3]'),
        ]:
            with self.subTest(schema=schema):
                prepared = structured.prepare_response_format(request_format(schema))
                prepared.validate_content(valid)
                with self.assertRaises(structured.StructuredOutputValidationError):
                    prepared.validate_content(invalid)

    def test_plain_json_and_text_behavior(self):
        prepared = structured.prepare_response_format({"type": "json_object"})
        prepared.validate_content('{"anything":1}')
        self.assertIsNone(structured.prepare_response_format({"type": "text"}))
        self.assertIsNone(structured.prepare_response_format(None))
        with self.assertRaises(structured.StructuredOutputValidationError):
            prepared.validate_content("not JSON")

    def test_duplicate_keys_non_finite_and_truncation_fail(self):
        prepared = structured.prepare_response_format({"type": "json_object"})
        for content in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '{"a":']:
            with self.subTest(content=content), self.assertRaises(structured.StructuredOutputValidationError):
                prepared.validate_content(content)
        with self.assertRaises(structured.StructuredOutputValidationError):
            prepared.validate_content('{"a":1}', "length")

    def test_valid_result_preserves_envelope(self):
        prepared = structured.prepare_response_format(request_format(PERSON))
        result = envelope('{"age":3}')
        self.assertIs(structured.validate_chat_result(result, prepared, stream=False), result)
        self.assertNotIn("refusal", result["choices"][0]["message"])

    def test_refusal_preserved_and_bad_envelope_rejected(self):
        prepared = structured.prepare_response_format(request_format(PERSON))
        refusal = envelope(None, refusal="Cannot answer")
        self.assertIs(structured.validate_chat_result(refusal, prepared, stream=False), refusal)
        for result in [{}, {"choices": []}, envelope('{"age":-1}'), envelope(None)]:
            with self.subTest(result=result), self.assertRaises(structured.StructuredOutputValidationError):
                structured.validate_chat_result(result, prepared, stream=False)

    def test_valid_stream_preserves_chunks_and_usage(self):
        prepared = structured.prepare_response_format(request_format(PERSON))
        source = list(chunks('{"age":3}')) + [{"choices": [], "usage": {"completion_tokens": 5}}]
        self.assertEqual(list(structured.validate_chat_result(iter(source), prepared, stream=True)), source)

    def test_invalid_stream_withholds_successful_terminal_chunk(self):
        prepared = structured.prepare_response_format(request_format(PERSON))
        source = list(chunks('{"age":-1}'))
        stream = structured.validate_chat_result(iter(source), prepared, stream=True)
        self.assertEqual(next(stream), source[0])
        with self.assertRaises(structured.StructuredOutputValidationError):
            next(stream)

    def test_missing_terminal_chunk_and_tool_stream_fail(self):
        prepared = structured.prepare_response_format(request_format(PERSON))
        for source in [list(chunks('{"age":3}'))[:1],
                       [{"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "x"}]}, "finish_reason": None}]}]]:
            with self.subTest(source=source), self.assertRaises(structured.StructuredOutputValidationError):
                list(structured.validate_chat_result(iter(source), prepared, stream=True))

    def test_stream_closes_generator_on_failure(self):
        closed = []

        def source():
            try:
                yield from chunks('{"age":-1}')
            finally:
                closed.append(True)

        prepared = structured.prepare_response_format(request_format(PERSON))
        with self.assertRaises(structured.StructuredOutputValidationError):
            list(structured.validate_chat_result(source(), prepared, stream=True))
        self.assertEqual(closed, [True])

    def test_central_method_validates_custom_handler_output(self):
        model = FakeLlama(lambda **kwargs: envelope('{"age":-1}'))
        with self.assertRaises(structured.StructuredOutputValidationError):
            model.create_chat_completion(messages=[], response_format=request_format(PERSON))

    def test_central_method_rejects_bad_request_before_handler(self):
        calls = []
        model = FakeLlama(lambda **kwargs: calls.append(kwargs))
        with self.assertRaises(structured.StructuredOutputRequestError):
            model.create_chat_completion(messages=[], response_format={"type": "json_schema"})
        self.assertEqual(calls, [])

    def test_unstructured_handler_behavior_is_unchanged(self):
        response = envelope("plain text")
        model = FakeLlama(lambda **kwargs: response)
        self.assertIs(model.create_chat_completion(messages=[]), response)

    def test_conflicting_grammar_or_tools_fail(self):
        prepared = structured.prepare_response_format(request_format(PERSON))
        for options in [{"grammar": "root ::= ..."}, {"tools": [{"type": "function"}]},
                        {"functions": [{"name": "function"}]}]:
            with self.subTest(options=options), self.assertRaises(structured.StructuredOutputRequestError):
                structured.check_structured_output_options(prepared, **options)
        structured.check_structured_output_options(prepared, tools=[{}], tool_choice="none")

    def test_request_serialization_preserves_legacy_schema_and_optional_fields(self):
        body = server_types.CreateChatCompletionRequest(
            messages=[{"role": "assistant", "content": "prior answer"}],
            response_format={"type": "json_object", "schema": PERSON, "strict": True},
        )
        value = body.model_dump()
        self.assertEqual(value["response_format"]["schema"], PERSON)
        self.assertNotIn("name", value["messages"][0])

    def test_http_nested_and_legacy_schema_requests(self):
        from fastapi.testclient import TestClient
        for response_format in [request_format(PERSON), {"type": "json_object", "schema": PERSON}]:
            with self.subTest(response_format=response_format):
                with TestClient(http_app(lambda **kwargs: envelope('{"age":3}'))) as client:
                    response = client.post("/v1/chat/completions", json={"messages": [], "response_format": response_format})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(json.loads(response.json()["choices"][0]["message"]["content"]), {"age": 3})

    def test_http_invalid_schema_returns_400_before_inference(self):
        from fastapi.testclient import TestClient
        calls = []
        with TestClient(http_app(lambda **kwargs: calls.append(kwargs))) as client:
            response = client.post("/v1/chat/completions", json={"messages": [], "stream": True,
                                   "response_format": request_format({"type": "number", "minimum": 0.5})})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["code"], "invalid_response_format")
        self.assertEqual(calls, [])

    def test_http_bad_generated_output_returns_422(self):
        from fastapi.testclient import TestClient
        with TestClient(http_app(lambda **kwargs: envelope('{"age":-1}'))) as client:
            response = client.post("/v1/chat/completions", json={"messages": [], "response_format": request_format(PERSON)})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["error"]["code"], "structured_output_validation_failed")

    def test_http_stream_success_and_validation_error(self):
        from fastapi.testclient import TestClient
        for content, success in [('{"age":3}', True), ('{"age":-1}', False)]:
            with self.subTest(content=content):
                with TestClient(http_app(lambda **kwargs: chunks(content))) as client:
                    response = client.post("/v1/chat/completions", json={"messages": [], "stream": True,
                                           "response_format": request_format(PERSON)})
                self.assertEqual(response.status_code, 200)
                if success:
                    self.assertIn("[DONE]", response.text)
                else:
                    self.assertIn("structured_output_validation_failed", response.text)
                    self.assertNotIn("[DONE]", response.text)

    def test_http_stream_releases_receive_channel(self):
        from fastapi.testclient import TestClient
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always", ResourceWarning)
            with TestClient(http_app(lambda **kwargs: chunks('{"age":3}'))) as client:
                response = client.post("/v1/chat/completions", json={"messages": [], "stream": True,
                                       "response_format": request_format(PERSON)})
                self.assertEqual(response.status_code, 200)
            gc.collect()
        self.assertFalse([warning for warning in recorded
                          if issubclass(warning.category, ResourceWarning) and "MemoryObject" in str(warning.message)])

    @unittest.skipUnless(importlib.util.find_spec("instructor"), "optional Instructor integration dependency")
    def test_actual_instructor_dynamic_models_and_business_validation_retry(self):
        import instructor
        import openai
        from fastapi.testclient import TestClient
        from pydantic import BaseModel, Field, field_validator

        class AgeAnswer(BaseModel):
            age: int = Field(ge=0)

            @field_validator("age")
            @classmethod
            def even_age(cls, value):
                if value % 2:
                    raise ValueError("Age must be even for this business rule.")
                return value

        class LabelAnswer(BaseModel):
            label: str

        calls = []

        def handler(**kwargs):
            schema = kwargs["response_format"]["json_schema"]["schema"]
            calls.append(schema)
            if "age" in schema.get("properties", {}):
                return envelope('{"age":3}' if len(calls) == 1 else '{"age":4}')
            return envelope('{"label":"dynamic"}')

        with TestClient(http_app(handler)) as transport:
            sdk = openai.OpenAI(base_url="http://testserver/v1", api_key="unit-test", http_client=transport)
            client = instructor.from_openai(sdk, mode=instructor.Mode.JSON_SCHEMA)
            result = client.chat.completions.create(model="test", response_model=AgeAnswer,
                messages=[{"role": "user", "content": "Return an age"}], max_retries=2)
            self.assertEqual(result.age, 4)
            self.assertEqual(len(calls), 2)
            result = client.chat.completions.create(model="test", response_model=LabelAnswer,
                messages=[{"role": "user", "content": "Return a label"}], max_retries=1)
            self.assertEqual(result.label, "dynamic")
            self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
