# Reticulate Protocol Check — GitHub Action

Check client code conformance against session type protocols in CI.

## Usage

```yaml
- name: Check Iterator conformance
  uses: bica-tools/reticulate/github-action@main
  with:
    protocol: java_iterator
    trace-file: tests/traces/iterator_traces.csv

- name: Check custom protocol
  uses: bica-tools/reticulate/github-action@main
  with:
    session-type: 'rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}'
    trace: 'hasNext,TRUE,next,hasNext,FALSE'
```

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| `session-type` | One of session-type or protocol | Session type string |
| `protocol` | One of session-type or protocol | Named protocol (e.g., `java_iterator`) |
| `trace-file` | One of trace-file or trace | File with traces (one per line, comma-separated) |
| `trace` | One of trace-file or trace | Single comma-separated trace |

## Outputs

| Output | Description |
|--------|-------------|
| `conforming` | Number of conforming traces |
| `violating` | Number of violating traces |
| `result-json` | Full JSON result |

## Available Protocols

- `java_iterator` — java.util.Iterator
- `java_inputstream` — java.io.InputStream  
- `jdbc_connection` — java.sql.Connection
- `python_file` — Python file object
- `kafka_producer` — Apache Kafka Producer
- `mongodb` — MongoDB Client

## Trace Format

One trace per line, methods and selection outcomes comma-separated:

```
hasNext,TRUE,next,hasNext,TRUE,next,hasNext,FALSE
hasNext,FALSE
read,data,read,EOF,close
```
