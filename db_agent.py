"""
LangGraph agent that assembles a SQLAlchemy URI from user-provided *pieces*
(“specs”), connects using langchain_community.SQLDatabase, and then exposes
safe read-only SQL tools. Works for any SQLAlchemy dialect supported by
SQLDatabase (Postgres, MySQL/MariaDB, SQL Server via pyodbc, SQLite, DuckDB,
Oracle, Snowflake*, etc.) as long as the required driver is installed.

*Note*: Snowflake/Oracle require their SQLAlchemy dialect packages installed.

USAGE (chat flow examples)
-------------------------
User: Dialect is mssql, driver is ODBC Driver 18 for SQL Server.
User: Host 10.0.0.5, port 1433. Database Reporting. Username report_ro. Password ***.
User: Params TrustServerCertificate=yes, Encrypt=no.
User: Allowed tables: Customers, Orders, OrderDetails.
User: Build and connect.
User: Who are the top 5 customers by total order amount?

Another example (Postgres):
User: Dialect postgresql+psycopg2, host db.internal, port 5432, db analytics.
User: Username analytics_ro, password ***.
User: Build and connect. Then: show daily order counts last 7 days.

DuckDB file:
User: Dialect duckdb, database /data/warehouse.duckdb. Build and connect. Then: list tables.

----------------------------------
Security & Guardrails
- Read-only: blocks INSERT/UPDATE/DELETE/DDL.
- Optional whitelist via allowed_tables.
- Auto-LIMIT on queries that return rows and forgot LIMIT/TOP.
- Passwords masked in logs/echo.

----------------------------------
Dependencies
pip install -U langgraph langchain_community "langchain[openai]" sqlparse
# plus the DB drivers you need, e.g.:
#   Postgres: psycopg2-binary
#   MySQL/MariaDB: pymysql or mysqlclient
#   SQL Server: pyodbc (and corresponding ODBC driver installed)
#   DuckDB: duckdb
#   Oracle: oracledb
#   Snowflake: snowflake-sqlalchemy
"""
from __future__ import annotations

from typing import Optional, TypedDict, List, Dict, Any
import urllib.parse
import sqlparse

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDataBaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Global state (simple in-memory)
# ------------------------------
class GraphState(TypedDict):
    specs: Dict[str, Any]           # collected pieces
    uri: Optional[str]
    connected: bool
    allowed_tables: Optional[List[str]]

state: GraphState = {"specs": {}, "uri": None, "connected": False, "allowed_tables": None}
_db_ref: Dict[str, Optional[SQLDatabase]] = {"db": None}

# ------------------------------
# Helpers: URI assembly & safety
# ------------------------------
# Template helpers for popular dialects. You can still pass a full custom
# dialect string (e.g., "postgresql+psycopg2", "mssql+pyodbc").
SUPPORTED_TEMPLATES = {
    # engine           : required fields
    "postgresql+psycopg2": ["username", "password", "host", "port", "database"],
    "postgresql": ["username", "password", "host", "port", "database"],
    "mysql+pymysql": ["username", "password", "host", "port", "database"],
    "mysql": ["username", "password", "host", "port", "database"],
    "mariadb+pymysql": ["username", "password", "host", "port", "database"],
    "mssql+pyodbc": ["username", "password", "host", "port", "database", "driver"],
    "sqlite": ["database"],            # file path or ":memory:"
    "duckdb": ["database"],            # file path or ":memory:"
    "oracle+oracledb": ["username", "password", "host", "port", "service_name"],
    "snowflake": ["username", "password", "account", "database", "schema"],  # plus optional warehouse, role
}

ALIAS_TO_DIALECT = {
    # convenient aliases a user might give
    "postgres": "postgresql+psycopg2",
    "postgresql": "postgresql+psycopg2",
    "pg": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "mariadb": "mariadb+pymysql",
    "mssql": "mssql+pyodbc",
    "sqlserver": "mssql+pyodbc",
    "sqlite": "sqlite",
    "duckdb": "duckdb",
    "oracle": "oracle+oracledb",
    "snowflake": "snowflake",
}

SENSITIVE_KEYS = {"password", "secret", "token"}


def _mask(v: Any) -> Any:
    if isinstance(v, str) and v and any(k in v.lower() for k in ["password=", "pwd="]):
        return "***"
    return "***" if isinstance(v, str) and v and v.lower() in SENSITIVE_KEYS else v


def _normalize_dialect(dialect: str) -> str:
    d = dialect.strip()
    return ALIAS_TO_DIALECT.get(d.lower(), d)


def _needs(key: str, specs: Dict[str, Any]) -> bool:
    return key not in specs or specs.get(key) in (None, "", [])


def _assemble_uri(specs: Dict[str, Any]) -> str:
    """Create a SQLAlchemy URI from collected specs. Raises ValueError if missing pieces."""
    dialect = _normalize_dialect(specs.get("dialect", ""))
    if not dialect:
        raise ValueError("Missing 'dialect'.")

    # Determine required fields
    required = SUPPORTED_TEMPLATES.get(dialect)
    if required is None:
        # Fall back to generic pattern requiring at least host/db or file-based
        required = []  # be permissive; user gives full custom path

    # Validate requirements
    missing = [k for k in required if _needs(k, specs)]
    if missing:
        raise ValueError(f"Missing required spec(s) for {dialect}: {', '.join(missing)}")

    params = specs.get("params") or {}

    if dialect.startswith("sqlite"):
        database = specs["database"]
        if database == ":memory:":
            return "sqlite:///:memory:"
        # absolute path recommended: sqlite:////absolute/path.db
        path = database if database.startswith("/") else database
        return f"sqlite:///{path}"

    if dialect.startswith("duckdb"):
        database = specs["database"]
        if database == ":memory:":
            return "duckdb:///:memory:"
        return f"duckdb:///{database}"

    if dialect.startswith("snowflake"):
        # snowflake://<user>:<password>@<account>/<database>/<schema>?warehouse=<warehouse>&role=<role>
        user = urllib.parse.quote(specs["username"]) if "username" in specs else ""
        pwd = urllib.parse.quote(specs.get("password", ""))
        account = specs["account"]
        database = specs["database"]
        schema = specs["schema"]
        q = params.copy()
        warehouse = specs.get("warehouse")
        role = specs.get("role")
        if warehouse: q["warehouse"] = warehouse
        if role: q["role"] = role
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        auth = f"{user}:{pwd}@" if user else ""
        return f"snowflake://{auth}{account}/{database}/{schema}{qs}"

    if dialect.startswith("oracle"):
        # oracle+oracledb://user:pass@host:port/?service_name=...
        user = urllib.parse.quote(specs["username"]) if "username" in specs else ""
        pwd = urllib.parse.quote(specs.get("password", ""))
        host = specs["host"]
        port = specs.get("port", 1521)
        q = {"service_name": specs["service_name"]}
        q.update(params)
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        auth = f"{user}:{pwd}@" if user else ""
        return f"oracle+oracledb://{auth}{host}:{port}/{qs}"

    if dialect.startswith("mssql+pyodbc"):
        # mssql+pyodbc://user:pass@host,port/db?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
        user = urllib.parse.quote(specs.get("username", ""))
        pwd = urllib.parse.quote(specs.get("password", ""))
        host = specs.get("host", "")
        port = specs.get("port")
        server = f"{host},{port}" if port else host
        database = specs.get("database", "")
        q = params.copy()
        if "driver" in specs:
            q["driver"] = specs["driver"]
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        auth = f"{user}:{pwd}@" if user else ""
        return f"mssql+pyodbc://{auth}{server}/{database}{qs}"

    # Generic scheme: dialect://user:pass@host:port/db?key=val
    user = urllib.parse.quote(specs.get("username", ""))
    pwd = urllib.parse.quote(specs.get("password", ""))
    auth = f"{user}:{pwd}@" if user else ""
    host = specs.get("host", "")
    port = f":{specs['port']}" if specs.get("port") else ""
    database = specs.get("database", "")
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    return f"{dialect}://{auth}{host}{port}/{database}{qs}"


# ------------------------------
# Tooling
# ------------------------------
@tool
def set_spec(key: str, value: Any) -> str:
    """Set a specification piece, e.g., key="host", value="db.internal".
    Keys: dialect, driver, host, port, database, username, password,
    account (Snowflake), schema (Snowflake), warehouse, role,
    service_name (Oracle), params (dict), allowed_tables (list of str).
    """
    if key == "params" and not isinstance(value, dict):
        return "For 'params', provide a JSON object (dict)."
    if key == "allowed_tables" and not isinstance(value, list):
        return "For 'allowed_tables', provide a JSON array of table names."
    state["specs"][key] = value
    if key == "allowed_tables":
        state["allowed_tables"] = value
    safe = {k: ("***" if k in SENSITIVE_KEYS or k == "password" else _mask(v)) for k, v in state["specs"].items()}
    return f"Spec set. Current specs (masked): {safe}"


@tool
def show_missing() -> str:
    """Show currently missing fields for the chosen dialect."""
    specs = state["specs"]
    if not specs.get("dialect"):
        return "Missing: dialect"
    dialect = _normalize_dialect(specs["dialect"])
    required = SUPPORTED_TEMPLATES.get(dialect)
    if required is None:
        return f"Dialect {dialect} not in templates. Provide fields as needed or a full custom dialect string."
    missing = [k for k in required if _needs(k, specs)]
    return "All required fields present." if not missing else f"Missing: {', '.join(missing)}"


@tool
def build_uri() -> str:
    """Build the SQLAlchemy URI from collected specs and store it for connection."""
    try:
        uri = _assemble_uri(state["specs"])
        state["uri"] = uri
        masked = uri
        # hide password in echo by masking in-place
        if ":" in masked and "@" in masked:
            before_at = masked.split("@", 1)[0]
            if ":" in before_at:
                user_part, rest = before_at.split(":", 1)
                if rest:
                    masked = masked.replace(f":{rest}@", ":***@")
        return f"URI built: {masked}"
    except Exception as e:
        return f"Could not build URI: {e}"


def _assert_connected():
    if not state["connected"] or _db_ref["db"] is None:
        raise RuntimeError("Not connected. Use build_uri then connect_db.")


def _is_read_only(sql: str) -> bool:
    banned = {"insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"}
    parsed = sqlparse.parse(sql)
    if not parsed:
        return True
    tokens = [t.value.lower() for t in parsed[0].flatten()]
    text = " ".join(tokens)
    return not any(kw in text for kw in banned)


@tool
def connect_db() -> str:
    """Connect using the previously built URI and optional allowed_tables."""
    if not state.get("uri"):
        return "No URI yet. Call build_uri first."
    try:
        include = state.get("allowed_tables")
        db = SQLDatabase.from_uri(state["uri"], include_tables=include)
        # sanity call
        _ = db.get_usable_table_names()
        _db_ref["db"] = db
        state["connected"] = True
        return f"Connected. Allowed tables: {include or 'ALL (DB perms apply)'}"
    except Exception as e:
        return f"Connection failed: {e}"


@tool
def connect_with_specs(specs: dict) -> str:
    """Spesleri tek adımda alır, URI'yi oluşturur ve bağlanır.
    Örnek:
    {
      "dialect": "mssql+pyodbc",
      "host": "10.0.0.5", "port": 1433, "database": "Reporting",
      "username": "report_ro", "password": "…",
      "params": {"driver":"ODBC Driver 18 for SQL Server",
                 "TrustServerCertificate":"yes","Encrypt":"no"},
      "allowed_tables":["Customers","Orders","OrderDetails"]
    }
    """
    # 1) specs -> URI
    state["specs"] = specs
    uri_msg = build_uri()
    if not state.get("uri"):
        return f"URI oluşturulamadı: {uri_msg}"
    # 2) connect
    conn_msg = connect_db()
    return f"{uri_msg} {conn_msg}"


@tool
def list_sql_database_tables() -> str:
    """List usable table names from the active connection."""
    _assert_connected()
    names = sorted(_db_ref["db"].get_usable_table_names())
    return "".join(names) if names else "(no usable tables found)"


@tool
def info_sql_database(table_names: str) -> str:
    """Return schema info for comma-separated table names."""
    _assert_connected()
    include = state.get("allowed_tables")
    req = [t.strip() for t in table_names.split(",") if t.strip()]
    if include:
        not_allowed = [t for t in req if t not in include]
        if not_allowed:
            return f"Table(s) not allowed: {', '.join(not_allowed)}. Allowed: {', '.join(include)}"
    # SQLDatabase.get_table_info expects comma-separated string
    return _db_ref["db"].get_table_info(", ".join(req))


@tool
def query_sql_checker(query: str) -> str:
    """LLM-based SQL sanity check (read-only enforced)."""
    if not _is_read_only(query):
        return "❌ BLOCKED: Only read-only queries are permitted."
    _assert_connected()
    checker = QuerySQLCheckerTool(db=_db_ref["db"], llm=init_chat_model("openai:gpt-4.1-mini"))
    return checker(query)


@tool
def query_sql_db(query: str) -> str:
    """Execute a read-only SQL query on the active connection. Auto-LIMIT if missing."""
    _assert_connected()
    if not _is_read_only(query):
        return "❌ BLOCKED: Only read-only queries are permitted."
    ql = query.lower()
    if "select" in ql and "count(" not in ql and " limit " not in ql and " top " not in ql:
        dialect_name = getattr(_db_ref["db"], "dialect", "").name if hasattr(_db_ref["db"], "dialect") else ""
        if dialect_name in {"postgresql", "mysql", "sqlite", "duckdb"}:
            query = query.rstrip("; ") + " LIMIT 50"
        else:
            if ql.strip().startswith("select "):
                query = query.replace("select ", "SELECT TOP 50 ", 1)
    return _db_ref["db"].run(query)


# ------------------------------
# Build the agent
# ------------------------------
SYSTEM_PROMPT = """
You are a READ-ONLY SQL assistant that first collects DB specs (dialect, host,
port, database, username, password, params), builds a URI, connects, then
answers questions using a strict tool sequence.

MANDATORY PLAN:
1) If no dialect → ask or infer; call set_spec("dialect", <value>). Prefer canonical forms: mssql+pyodbc, postgresql+psycopg2, mysql+pymysql, duckdb, sqlite, oracle+oracledb, snowflake.
2) Call show_missing and fill pieces with set_spec(...) until it reports "All required fields present." If user already provided everything, you MAY call connect_with_specs(specs) directly.
3) If a whitelist is provided, set set_spec("allowed_tables", [...]).
4) Build & connect: EITHER (a) build_uri → connect_db, OR (b) connect_with_specs(specs). DO NOT stop after printing a URI; you MUST establish a connection.
5) list_sql_database_tables → info_sql_database for relevant tables → query_sql_checker → query_sql_db.
6) Never run DML/DDL. Prefer aggregates. Keep outputs concise and mention which tables were used.
"""

TOOLS = [
    set_spec,
    show_missing,
    build_uri,
    connect_with_specs,
    connect_db,
    list_sql_database_tables,
    info_sql_database,
    query_sql_checker,
    query_sql_db,
]

LLM = init_chat_model("openai:gpt-4.1")
AGENT = create_react_agent(LLM, TOOLS, prompt=SYSTEM_PROMPT)
checkpointer = MemorySaver()

AGENT = create_react_agent(
    LLM, TOOLS, prompt=SYSTEM_PROMPT, checkpointer=checkpointer, debug=False
)

async def bot_answer(message, conversation_id, username):
    config = {"configurable": {"thread_id": conversation_id}, "recursion_limit": 25}
    # Only the new message – no manual history
    response = await AGENT.ainvoke({"messages": [("user", message)]}, config=config)
    return response["messages"][-1].content

# ------------------------------
# Example runner (optional)
# ------------------------------
if __name__ == "__main__":
    # Simple interactive example (SQL Server). Replace with your own specs.
    messages = [
        ("user", "Dialect is mssql. Driver is ODBC Driver 18 for SQL Server."),
        ("user", "Host 127.0.0.1, port 1433, database Reporting."),
        ("user", "Username readonly, password p@ssw0rd."),
        ("user", "Params TrustServerCertificate=yes, Encrypt=no."),
        ("user", "Allowed tables: Customers, Orders, OrderDetails."),
        ("user", "Build and connect; then show top 5 customers by total order amount."),
    ]
    result = AGENT.invoke({"messages": messages})
    print(result["messages"][-1].content)
