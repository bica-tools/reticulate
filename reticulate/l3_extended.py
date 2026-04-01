"""Extended L3 protocol catalog — 60+ additional real-world protocols.

Augments the 16 protocols in l3_protocols.py and the 108 benchmarks
to provide statistical confidence (150+ total L3 protocols).

Organized by domain:
  - Java stdlib (15 protocols)
  - Python stdlib (10 protocols)
  - Database clients (8 protocols)
  - Messaging/streaming (7 protocols)
  - Cloud/infrastructure (5 protocols)
  - Network/security (8 protocols)
  - UI/interactive (7 protocols)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Extended L3 Protocol Catalog
# ---------------------------------------------------------------------------

EXTENDED_L3: dict[str, str] = {

    # =======================================================================
    # Java stdlib — additional protocols
    # =======================================================================

    # java.util.Scanner
    "java_scanner": "rec X . &{hasNext: +{TRUE: &{next: X, nextInt: X, nextLine: X}, FALSE: &{close: end}}}",

    # java.util.ListIterator (bidirectional)
    "java_list_iterator": "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}, hasPrevious: +{TRUE: &{previous: X}, FALSE: end}}",

    # java.io.RandomAccessFile
    "java_random_access": "&{open: rec X . &{read: +{data: X, EOF: &{close: end}}, write: X, seek: X, getFilePointer: X, close: end}}",

    # java.nio.channels.SocketChannel
    "java_nio_socket": "&{open: &{connect: +{OK: rec X . &{read: +{bytes: X, EOF: &{close: end}}, write: +{OK: X, ERR: X}, close: end}, FAIL: end}}}",

    # java.util.concurrent.BlockingQueue
    "java_blocking_queue": "&{create: rec X . &{put: X, offer: +{TRUE: X, FALSE: X}, take: X, poll: +{ITEM: X, NULL: X}, close: end}}",

    # java.util.concurrent.Future
    "java_future": "&{submit: &{isDone: +{TRUE: &{get: +{RESULT: end, EXCEPTION: end}}, FALSE: &{cancel: +{TRUE: end, FALSE: end}}}}}",

    # java.util.concurrent.CompletableFuture
    "java_completable_future": "&{supplyAsync: +{COMPLETE: &{thenApply: +{COMPLETE: &{get: end}, EXCEPTION: end}}, EXCEPTION: &{exceptionally: end}}}",

    # java.util.stream.Stream
    "java_stream": "&{stream: &{filter: &{map: &{collect: end}}}}",

    # java.util.Optional
    "java_optional": "&{of: +{PRESENT: &{get: end, map: +{PRESENT: end, EMPTY: end}}, EMPTY: &{orElse: end}}}",

    # javax.xml.parsers.SAXParser
    "java_sax_parser": "&{parse: rec X . +{startElement: X, endElement: X, characters: X, endDocument: end, error: end}}",

    # java.security.MessageDigest
    "java_message_digest": "&{getInstance: rec X . &{update: X, digest: end, reset: X}}",

    # java.util.zip.ZipInputStream
    "java_zip_input": "&{open: rec X . &{getNextEntry: +{ENTRY: &{read: +{data: X, EOF: &{closeEntry: X}}, closeEntry: X}, NULL: &{close: end}}}}",

    # java.nio.file.DirectoryStream
    "java_dir_stream": "&{open: rec X . &{iterator: &{hasNext: +{TRUE: &{next: X}, FALSE: &{close: end}}}}}",

    # javax.net.ssl.SSLSocket
    "java_ssl_socket": "&{connect: &{startHandshake: +{OK: rec X . &{read: +{data: X, EOF: &{close: end}}, write: X, close: end}, FAIL: end}}}",

    # java.sql.PreparedStatement
    "java_prepared_stmt": "&{prepare: rec X . &{setParam: X, executeQuery: &{next: +{TRUE: &{get: X}, FALSE: &{close: end}}}, executeUpdate: +{OK: &{close: end}, ERR: &{close: end}}}}",

    # =======================================================================
    # Python stdlib — additional protocols
    # =======================================================================

    # asyncio.StreamReader/StreamWriter
    "python_asyncio_stream": "&{open_connection: rec X . &{read: +{data: X, EOF: &{close: end}}, write: &{drain: X}, close: end}}",

    # ssl.SSLSocket
    "python_ssl": "&{wrap_socket: +{OK: rec X . &{read: +{data: X, EOF: &{close: end}}, write: X, close: end}, FAIL: end}}",

    # ftplib.FTP
    "python_ftp": "&{connect: &{login: +{OK: rec X . &{cwd: X, list: X, retrbinary: +{OK: X, ERR: X}, storbinary: +{OK: X, ERR: X}, quit: end}, FAIL: end}}}",

    # smtplib.SMTP (Python)
    "python_smtp": "&{connect: &{ehlo: +{OK: &{login: +{OK: rec X . &{sendmail: +{OK: X, ERR: X}, quit: end}, FAIL: end}}, FAIL: end}}}",

    # tarfile.TarFile
    "python_tarfile": "&{open: rec X . &{next: +{MEMBER: &{extractfile: +{data: X, ERR: X}}, NULL: &{close: end}}, close: end}}",

    # xml.etree.ElementTree iterparse
    "python_iterparse": "&{iterparse: rec X . +{start: &{process: X}, end: &{process: X}, done: end}}",

    # multiprocessing.Pool
    "python_pool": "&{create: rec X . &{apply_async: X, map: X, close: &{join: end}, terminate: end}}",

    # subprocess.Popen
    "python_popen": "&{open: &{communicate: +{OK: &{wait: end}, TIMEOUT: &{kill: &{wait: end}}}}}",

    # tempfile.NamedTemporaryFile
    "python_tempfile": "&{create: rec X . &{write: X, read: X, seek: X, close: &{cleanup: end}}}",

    # logging.Handler
    "python_logging_handler": "&{create: rec X . &{emit: X, flush: X, close: end}}",

    # =======================================================================
    # Database clients
    # =======================================================================

    # MongoDB client
    "mongodb_client": "&{connect: &{getDatabase: &{getCollection: rec X . &{insertOne: +{OK: X, ERR: X}, find: &{next: +{DOC: X, EXHAUSTED: X}}, deleteOne: +{OK: X, ERR: X}, close: end}}}}",

    # Redis client
    "redis_client": "&{connect: rec X . &{get: +{VALUE: X, NIL: X}, set: +{OK: X, ERR: X}, del: +{OK: X, ERR: X}, expire: +{OK: X, ERR: X}, ping: +{PONG: X}, quit: end}}",

    # PostgreSQL (psycopg2)
    "postgres_client": "&{connect: &{cursor: rec X . &{execute: +{OK: &{fetchone: +{ROW: X, NONE: X}, fetchall: X}, ERR: X}, commit: X, rollback: X, close: &{close: end}}}}",

    # Elasticsearch client
    "elasticsearch_client": "&{connect: rec X . &{index: +{OK: X, ERR: X}, search: +{HITS: X, ERR: X}, delete: +{OK: X, ERR: X}, close: end}}",

    # Apache Cassandra
    "cassandra_client": "&{connect: &{prepare: rec X . &{bind: &{execute: +{ROWS: X, ERR: X}}, close: &{close: end}}}}",

    # SQLite WAL mode
    "sqlite_wal": "&{connect: &{pragma_wal: +{OK: rec X . &{begin: &{execute: +{OK: &{commit: X, rollback: X}, ERR: &{rollback: X}}}, close: end}, ERR: end}}}",

    # DynamoDB client
    "dynamodb_client": "&{createTable: +{OK: rec X . &{putItem: +{OK: X, ERR: X}, getItem: +{FOUND: X, NOT_FOUND: X}, query: +{ITEMS: X, ERR: X}, deleteTable: end}, ERR: end}}",

    # LMDB (memory-mapped DB)
    "lmdb_client": "&{open_env: &{begin_txn: rec X . &{get: +{VALUE: X, NOT_FOUND: X}, put: X, delete: +{OK: X, NOT_FOUND: X}, commit: end, abort: end}}}",

    # =======================================================================
    # Messaging / streaming
    # =======================================================================

    # Apache Kafka producer
    "kafka_producer": "&{create: rec X . &{send: +{ACK: X, ERR: X}, flush: X, close: end}}",

    # Apache Kafka consumer
    "kafka_consumer": "&{subscribe: rec X . &{poll: +{RECORDS: &{process: X}, EMPTY: X, ERR: X}, commitSync: X, close: end}}",

    # RabbitMQ channel
    "rabbitmq_channel": "&{open: &{declare_queue: rec X . &{publish: +{ACK: X, NACK: X}, consume: +{MSG: &{ack: X, nack: X}, TIMEOUT: X}, close: end}}}",

    # MQTT client
    "mqtt_client": "&{connect: +{CONNACK: rec X . &{publish: +{PUBACK: X, ERR: X}, subscribe: +{SUBACK: X, ERR: X}, loop: +{MSG: X, NONE: X}, disconnect: end}, FAIL: end}}",

    # gRPC unary call
    "grpc_unary": "&{createChannel: &{newStub: rec X . &{call: +{OK: X, UNAVAILABLE: X, DEADLINE_EXCEEDED: X}, shutdown: end}}}",

    # WebSocket client
    "websocket_client": "&{connect: +{OPEN: rec X . &{send: X, recv: +{MSG: X, CLOSE: end, ERR: end}}, FAIL: end}}",

    # Server-Sent Events
    "sse_client": "&{connect: +{OK: rec X . +{EVENT: &{process: X}, RETRY: X, CLOSE: end}, FAIL: end}}",

    # =======================================================================
    # Cloud / infrastructure
    # =======================================================================

    # AWS S3 (proper lifecycle)
    "aws_s3_object": "&{createBucket: +{OK: rec X . &{putObject: +{OK: X, ERR: X}, getObject: +{OK: &{read: +{data: X, EOF: X}}, NOT_FOUND: X}, deleteObject: +{OK: X, ERR: X}, deleteBucket: end}, ERR: end}}",

    # Docker container lifecycle
    "docker_lifecycle": "&{create: +{OK: &{start: +{OK: rec X . &{logs: X, exec: +{OK: X, ERR: X}, stop: +{OK: &{remove: end}, TIMEOUT: &{kill: &{remove: end}}}}, FAIL: end}}, ERR: end}}",

    # Kubernetes pod lifecycle
    "k8s_pod": "&{create: +{PENDING: &{schedule: +{RUNNING: rec X . &{exec: +{OK: X, ERR: X}, logs: X, delete: +{TERMINATING: &{wait: end}, ERR: X}}, FAILED: end}}, ERR: end}}",

    # Terraform resource
    "terraform_resource": "&{plan: +{CHANGES: &{apply: +{OK: &{refresh: +{OK: end, DRIFT: &{plan: +{CHANGES: end, NO_CHANGES: end}}}}, ERR: end}}, NO_CHANGES: end}}",

    # CI/CD pipeline stage
    "cicd_stage": "&{trigger: +{QUEUED: &{run: +{PASS: &{deploy: +{OK: end, ROLLBACK: end}}, FAIL: &{retry: +{PASS: end, FAIL: end}}}}, SKIP: end}}",

    # =======================================================================
    # Network / security
    # =======================================================================

    # HTTP/2 stream
    "http2_stream": "&{sendHeaders: +{CONTINUE: &{sendData: +{OK: &{recvHeaders: &{recvData: +{DATA: end, TRAILERS: end}}}, ERR: end}}, REFUSED: end}}",

    # QUIC connection
    "quic_connection": "&{connect: +{HANDSHAKE_OK: rec X . &{openStream: +{OK: &{send: +{ACK: X, LOST: X}}, ERR: X}, close: end}, FAIL: end}}",

    # OAuth2 refresh flow
    "oauth2_refresh": "&{authorize: +{CODE: &{exchange: +{TOKEN: rec X . &{useToken: +{OK: X, EXPIRED: &{refresh: +{NEW_TOKEN: X, REVOKED: end}}}, revoke: end}, ERR: end}}, DENIED: end}}",

    # LDAP bind/search
    "ldap_client": "&{connect: &{bind: +{OK: rec X . &{search: +{RESULTS: X, ERR: X}, modify: +{OK: X, ERR: X}, unbind: end}, FAIL: end}}}",

    # DNS recursive resolution
    "dns_recursive": "rec X . &{query: +{ANSWER: end, REFERRAL: &{followNS: X}, NXDOMAIN: end, SERVFAIL: &{retry: +{OK: X, TIMEOUT: end}}}}",

    # SAML authentication
    "saml_auth": "&{requestAuth: +{REDIRECT: &{submitCredentials: +{ASSERTION: &{validateSignature: +{VALID: end, INVALID: end}}, DENIED: end}}, ERR: end}}",

    # Certificate validation chain
    "cert_validation": "&{loadCert: &{checkExpiry: +{VALID: &{checkIssuer: +{TRUSTED: &{checkRevocation: +{OK: end, REVOKED: end}}, UNTRUSTED: &{fetchIntermediate: +{FOUND: end, NOT_FOUND: end}}}}, EXPIRED: end}}}",

    # Kerberos authentication
    "kerberos_auth": "&{requestTGT: +{OK: &{requestServiceTicket: +{OK: &{authenticate: +{OK: end, FAIL: end}}, ERR: end}}, ERR: end}}",

    # =======================================================================
    # UI / interactive
    # =======================================================================

    # Form wizard (multi-step)
    "form_wizard": "&{start: &{fillStep1: +{VALID: &{fillStep2: +{VALID: &{fillStep3: +{VALID: &{submit: +{OK: end, ERR: end}}, INVALID: end}}, INVALID: end}}, INVALID: end}}}",

    # Drag and drop
    "drag_drop": "&{mouseDown: &{dragStart: rec X . &{dragMove: X, dragEnd: +{DROP_OK: &{handleDrop: end}, DROP_CANCEL: end}}}}",

    # Modal dialog
    "modal_dialog": "&{open: &{display: rec X . &{interact: X, confirm: +{OK: &{close: end}, CANCEL: &{close: end}}}}}",

    # Undo/redo
    "undo_redo": "&{start: rec X . &{execute: X, canUndo: +{TRUE: &{undo: X}, FALSE: X}, canRedo: +{TRUE: &{redo: X}, FALSE: X}, quit: end}}",

    # Pagination
    "pagination": "&{firstPage: rec X . &{nextPage: +{OK: X, LAST: end}, prevPage: +{OK: X, FIRST: X}}}",

    # File upload
    "file_upload": "&{selectFile: +{OK: &{upload: +{PROGRESS: &{waitComplete: +{DONE: end, ERROR: end}}, ERROR: end}}, CANCEL: end}}",

    # Authentication flow (login/2FA)
    "login_2fa": "&{enterCredentials: +{OK: &{enter2FA: +{OK: end, INVALID: &{retry2FA: +{OK: end, LOCKED: end}}}}, INVALID: &{retryLogin: +{OK: end, LOCKED: end}}}}",
}
