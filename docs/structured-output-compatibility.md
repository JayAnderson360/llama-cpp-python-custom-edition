# Structured output compatibility

Custom edition accepts dynamic response schemas from Instructor and other
OpenAI-compatible clients. This does not depend on a particular Pydantic class
or on importing Instructor into the server.

Both request envelopes are supported, including through the HTTP request model:

```json
{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"age":{"type":"integer","minimum":0}},"required":["age"],"additionalProperties":false}}}
```

```json
{"type":"json_object","schema":{"type":"object","properties":{"age":{"type":"integer"}},"required":["age"]}}
```

Pass either object as `response_format`. A plain `{"type":"json_object"}`
request still uses generic JSON grammar. Text requests and ordinary tool-calling
requests without a JSON response format retain their previous behavior.

## Enforcement

- A schema is checked before inference and compiled to GBNF. Schema conversion
  failure never silently becomes generic JSON in the response-format path.
- Strict mode accepts a conservative subset of the bundled grammar converter.
  Unsupported keywords, ambiguous combinations, and unsupported numeric/string
  constraints are rejected before inference. This is not a claim of complete
  support for every JSON Schema dialect or constraint in constrained decoding.
- In non-strict mode, conversion must still succeed. Some constraints, such as
  decimal-number bounds, may be checked only on the completed output.
- Completed JSON is checked with `jsonschema`, including available format
  checkers. Duplicate keys, non-finite numbers, invalid JSON, schema mismatches,
  and incomplete generation are reported as errors. Validation errors do not
  include generated field values.
- Local JSON-pointer references are supported; external references, embedded
  resource scopes, and dynamic references are rejected. Literal JSON values in
  `const`, `enum`, defaults, or examples are not interpreted as schema references.
- A JSON response format cannot be combined with a separate grammar or active
  tool/function calling. Explicitly disabled tool/function calls are allowed.
- Explicit model refusals remain protocol refusals, not schema-valid answers.

The optional `name` and `refusal` message fields remain optional. The chat route
retains its flexible outer response serialization; structured responses receive
the shared message/JSON checks rather than the old restrictive response model.

## HTTP and streaming

Invalid or unsupported response formats produce an OpenAI-style HTTP **400**
error with code `invalid_response_format`. Generated output that fails validation
produces HTTP **422**, code `structured_output_validation_failed`.

Requests are checked before the SSE response starts. During streaming, content
chunks are provisional: the assembled JSON is validated before its successful
terminal chunk is emitted. A validation failure emits an SSE error payload and
does not emit `[DONE]`. Already emitted chunks cannot be recalled. Client
disconnects/interruption do not establish a successful structured result.

There are no hidden server retries. Keep Instructor/Pydantic validation for
custom Python validators and business rules. If retrying HTTP 422 responses is
desired, configure that in the client explicitly; Instructor's default parsing
retry policy should not be assumed to retry HTTP errors or streaming failures.

## Dependency and verification

Source environments need the new `jsonschema[format-nongpl]>=4.18.0` dependency.
Changing source files alone does not install it or rebuild an existing wheel.

`tests/test_structured_output.py` exercises the actual pure-Python implementation,
central chat method, HTTP route, request serialization, error handler, and SSE
publisher using a fake model. It needs `jsonschema[format-nongpl]`, `fastapi`,
`httpx`, and `sse-starlette`; it does not need native llama libraries, a GPU, or a
model. Run it on the designated testing machine:

```text
python -B tests/test_structured_output.py
```

Verification on PC 50.2 passed all 31 checks, including the actual Instructor
client using JSON_SCHEMA mode with two different Pydantic models and retrying a
custom business-validation failure. The HTTP transport and model were simulated;
the schema preparation, chat method, HTTP route, SDK parsing, and retry behavior
were exercised. The streaming receive-channel cleanup regression also passed.
The tested Python sources matched the local editing checkout byte for byte.

The isolated environment used Python 3.13 with Instructor 1.17.0, OpenAI SDK
3.3.0, Pydantic 2.14.3, jsonschema 4.26.0, FastAPI 0.163.2, and sse-starlette
3.7.0. No packages were installed during this verification run. Process-specific
temporary, user-data, and cache locations were redirected inside the testing
workspace, and Python bytecode writes were disabled.

These checks do not establish real-model generation, native grammar acceptance,
GPU behavior, or compatibility with every client version and chat handler.
Do not run them in the local editing workspace when the project's execution
policy assigns testing to another machine.
