"""
SQLGraphAgent: A Python class that
- builds a SQLAlchemy URI from explicit parameters,
- connects to the DB via langchain_community.SQLDatabase,
- creates a LangGraph ReAct agent with safe, read-only tools.

Works with any dialect supported by SQLDatabase (Postgres, MySQL/MariaDB,
SQL Server via pyodbc, SQLite, DuckDB, Oracle, Snowflake, ...), provided the
proper driver packages are installed.

pip install -U langgraph langchain_community "langchain[openai]" sqlparse
# plus DB drivers you actually need:
#   Postgres: psycopg2-binary
#   MySQL/MariaDB: pymysql or mysqlclient
#   SQL Server: pyodbc  (and OS-level ODBC Driver 18/17)
#   DuckDB: duckdb
#   Oracle: oracledb
#   Snowflake: snowflake-sqlalchemy
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Sequence
import urllib.parse
import sqlparse

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool, Tool
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# URI assembly helpers
# -----------------------------
ALIAS_TO_DIALECT = {
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

REQUIRED_FIELDS: Dict[str, Sequence[str]] = {
    "postgresql+psycopg2": ["username", "password", "host", "port", "database"],
    "mysql+pymysql": ["username", "password", "host", "port", "database"],
    "mariadb+pymysql": ["username", "password", "host", "port", "database"],
    "mssql+pyodbc": ["username", "password", "host", "port", "database", "driver"],
    "sqlite": ["database"],
    "duckdb": ["database"],
    "oracle+oracledb": ["username", "password", "host", "port", "service_name"],
    "snowflake": ["username", "password", "account", "database", "schema"],
}


def _normalize_dialect(dialect: str) -> str:
    return ALIAS_TO_DIALECT.get(dialect.lower(), dialect)


def build_sqlalchemy_uri(
    *,
    dialect: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    driver: Optional[str] = None,
    account: Optional[str] = None,    # snowflake
    schema: Optional[str] = None,     # snowflake
    warehouse: Optional[str] = None,  # snowflake
    role: Optional[str] = None,       # snowflake
    service_name: Optional[str] = None,  # oracle
) -> str:
    """Create a SQLAlchemy URI from explicit fields, with special cases for common dialects."""
    d = _normalize_dialect(dialect)
    params = params or {}

    # Validate required fields (best-effort)
    required = REQUIRED_FIELDS.get(d)
    if required:
        missing = []
        for key in required:
            if key == "driver" and d.startswith("mssql+pyodbc"):
                if not driver and "driver" not in params:
                    missing.append("driver")
            elif locals().get(key) in (None, ""):
                missing.append(key)
        if missing:
            raise ValueError(f"Missing required fields for {d}: {', '.join(missing)}")

    if d.startswith("sqlite"):
        assert database, "database (path or :memory:) is required for sqlite"
        return "sqlite:///:memory:" if database == ":memory:" else f"sqlite:///{database}"

    if d.startswith("duckdb"):
        assert database, "database (path or :memory:) is required for duckdb"
        return "duckdb:///:memory:" if database == ":memory:" else f"duckdb:///{database}"

    if d.startswith("snowflake"):
        assert account and database and schema, "account, database, schema required for snowflake"
        q = dict(params)
        if warehouse:
            q["warehouse"] = warehouse
        if role:
            q["role"] = role
        u = urllib.parse.quote(username or "") if username else ""
        p = urllib.parse.quote(password or "") if password else ""
        auth = f"{u}:{p}@" if username else ""
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        return f"snowflake://{auth}{account}/{database}/{schema}{qs}"

    if d.startswith("oracle+oracledb"):
        assert host and service_name, "host and service_name required for oracle+oracledb"
        q = {"service_name": service_name}
        q.update(params)
        u = urllib.parse.quote(username or "") if username else ""
        p = urllib.parse.quote(password or "") if password else ""
        auth = f"{u}:{p}@" if username else ""
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        return f"oracle+oracledb://{auth}{host}:{port or 1521}/{qs}"

    if d.startswith("mssql+pyodbc"):
        assert host and database, "host and database required for mssql+pyodbc"
        # host,port uses COMMA in SQL Server URIs
        server = f"{host},{port}" if port else host
        q = dict(params)
        if driver:
            q["driver"] = driver
        # Note: spaces must be '+' encoded in driver name, urlencode will handle it
        u = urllib.parse.quote(username or "") if username else ""
        p = urllib.parse.quote(password or "") if password else ""
        auth = f"{u}:{p}@" if username else ""
        qs = ("?" + urllib.parse.urlencode(q)) if q else ""
        return f"mssql+pyodbc://{auth}{server}/{database}{qs}"

    # Generic pattern
    u = urllib.parse.quote(username or "") if username else ""
    p = urllib.parse.quote(password or "") if password else ""
    auth = f"{u}:{p}@" if username else ""
    hostpart = host or ""
    portpart = f":{port}" if port else ""
    dbpart = database or ""
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    return f"{d}://{auth}{hostpart}{portpart}/{dbpart}{qs}"


# -----------------------------
# The class
# -----------------------------
@dataclass
class SQLGraphAgent:
    # required / common
    dialect: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    # extras by engine
    params: Dict[str, Any] = field(default_factory=dict)
    driver: Optional[str] = None              # mssql+pyodbc
    account: Optional[str] = None             # snowflake
    schema: Optional[str] = None              # snowflake
    warehouse: Optional[str] = None           # snowflake
    role: Optional[str] = None                # snowflake
    service_name: Optional[str] = None        # oracle

    # guardrails/settings
    allowed_tables: Optional[List[str]] = None
    readonly: bool = True
    default_row_limit: int = 50

    # LLM / agent
    model: str = "openai:gpt-4.1"
    checker_model: str = "openai:gpt-4.1-mini"

    # populated after init
    uri: str = field(init=False)
    db: SQLDatabase = field(init=False)
    tools: List[Tool] = field(init=False)
    agent: Any = field(init=False)

    def __post_init__(self):
        # 1) Build URI
        self.uri = build_sqlalchemy_uri(
            dialect=self.dialect,
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password,
            params=self.params,
            driver=self.driver,
            account=self.account,
            schema=self.schema,
            warehouse=self.warehouse,
            role=self.role,
            service_name=self.service_name,
        )
        # 2) Connect
        self.db = SQLDatabase.from_uri(self.uri, include_tables=self.allowed_tables)
        # sanity call
        _ = self.db.get_usable_table_names()
        # 3) Build tools & agent
        self.tools = self._build_tools()
        self.agent = self._build_agent()

    # -------------------------
    # Tools (as closures)
    # -------------------------
    def _is_read_only(self, sql: str) -> bool:
        if not self.readonly:
            return True
        banned = {"insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"}
        parsed = sqlparse.parse(sql)
        if not parsed:
            return True
        tokens = [t.value.lower() for t in parsed[0].flatten()]
        text = " ".join(tokens)
        return not any(kw in text for kw in banned)

    def _auto_limit(self, query: str) -> str:
        ql = query.lower()
        if "select" in ql and "count(" not in ql and " limit " not in ql and " top " not in ql:
            # Get dialect name safely
            dialect_name = ""
            if hasattr(self.db, "dialect"):
                dialect_obj = self.db.dialect
                if hasattr(dialect_obj, "name"):
                    dialect_name = dialect_obj.name
                elif isinstance(dialect_obj, str):
                    dialect_name = dialect_obj
            
            if dialect_name in {"postgresql", "mysql", "sqlite", "duckdb"}:
                return query.rstrip("; ") + f" LIMIT {self.default_row_limit}"
            else:
                # Assume SQL Server TOP
                if ql.strip().startswith("select "):
                    return query.replace("select ", f"SELECT TOP {self.default_row_limit} ", 1)
        return query

    def _build_tools(self) -> List[Tool]:
        llm_checker = init_chat_model(self.checker_model)

        @tool
        def list_sql_database_tables() -> str:
            """List usable table names from the active connection."""
            names = sorted(self.db.get_usable_table_names())
            return "\n".join(names) if names else "(no usable tables found)"

        @tool
        def info_sql_database(table_names: str) -> str:
            """Return schema info for comma-separated table names."""
            # Ensure we treat table_names as a single string, not an iterable
            if not table_names or not table_names.strip():
                return "No table names provided."
            
            # Split by comma and clean up whitespace
            req = [t.strip() for t in table_names.split(",") if t.strip()]
            
            if self.allowed_tables:
                not_allowed = [t for t in req if t not in (self.allowed_tables or [])]
                if not_allowed:
                    return (
                        f"Table(s) not allowed: {', '.join(not_allowed)}. "
                        f"Allowed: {', '.join(self.allowed_tables)}"
                    )
            
            # Pass the list of table names directly to get_table_info
            try:
                return self.db.get_table_info(req)
            except Exception as e:
                # Fallback: try with the original string format
                return self.db.get_table_info(table_names)

        @tool
        def query_sql_checker(query: str) -> str:
            """LLM-based SQL sanity check (read-only enforced)."""
            if not self._is_read_only(query):
                return "❌ BLOCKED: Only read-only queries are permitted."
            checker = QuerySQLCheckerTool(db=self.db, llm=llm_checker)
            return checker(query)

        @tool
        def query_sql_db(query: str) -> str:
            """Execute a read-only SQL query on the active connection. Auto-LIMIT if missing."""
            if not self._is_read_only(query):
                return "❌ BLOCKED: Only read-only queries are permitted."
            q = self._auto_limit(query)
            return self.db.run(q)

        return [
            list_sql_database_tables,
            info_sql_database,
            query_sql_checker,
            query_sql_db,
        ]

    # -------------------------
    # Agent creation & usage
    # -------------------------
    def _build_agent(self):
        system = f"""
You are a READ-ONLY SQL assistant for a {getattr(self.db, 'dialect', 'SQL')} database.
ALWAYS follow this sequence:
1) list_sql_database_tables → 2) info_sql_database (relevant tables) → 3) query_sql_checker → 4) query_sql_db.
Never run DML/DDL. Prefer aggregates. Keep outputs concise and mention which tables you used.
"""
        llm = init_chat_model(self.model)
        checkpointer = MemorySaver()

        return create_react_agent(llm, self.tools, prompt=system, checkpointer=checkpointer, debug=False)

    async def ask(self, user_message: str, conversation_id: str, username: str, as_dict: bool = False):
        """Send a message to the agent and return the final text (or full dict)."""
        config = {"configurable": {"thread_id": conversation_id}, "recursion_limit": 25}
        result = await self.agent.ainvoke({"messages": [("user", user_message)]}, config=config)
        return result if as_dict else result["messages"][-1].content


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # SQL Server example
    uri_params = dict(
        dialect="mssql+pyodbc",
        host="127.0.0.1",
        port=1433,
        database="Reporting",
        username="readonly",
        password="p@ssw0rd",
        driver="ODBC Driver 18 for SQL Server",
        params={"TrustServerCertificate": "yes", "Encrypt": "no"},
        allowed_tables=["Customers", "Orders", "OrderDetails"],
    )

    agent = SQLGraphAgent(**uri_params)

    print("\n-- TABLES --")
    print(agent.ask("List the tables"))

    print("\n-- QUESTION --")
    q = (
        "En çok sipariş tutarına sahip ilk 5 müşteri kim? "
        "(Customers, Orders, OrderDetails tablolarını kullan)"
    )
    print(agent.ask(q))
