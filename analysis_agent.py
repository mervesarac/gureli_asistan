import logging
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from quickchart import QuickChart

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

load_dotenv()

qc = QuickChart()
qc.width = 500
qc.height = 300
qc.config = {
    "type": "bar",
    "data": {
        "labels": ["Hello world", "Test"],
        "datasets": [{"label": "Foo", "data": [1, 2]}],
    },
}


@tool
def create_chart(
    width: int = 300,
    height: int = 500,
    chart_type: str = "bar",
    datasets_label: str = "",
    labels: list = None,
    data: list = None,
) -> str:
    """Create a chart using QuickChart and return the URL."""
    logging.info(
        f"Creating chart with width={width}, height={height}, chart_type={chart_type}, datasets_label={datasets_label}, labels={labels}, data={data}"
    )
    qc = QuickChart()
    # Set the chart width and height
    qc.width = width
    qc.height = height
    qc.config = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": [{"label": datasets_label, "data": data}],
        },
    }
    return qc.get_short_url()


def get_engine_for_db(url: str) -> create_engine:
    """Pull sql file, populate in-memory database, and create engine."""
    return create_engine(
        url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_db(
    "sqlite:////Users/mervesarac/Development/eguven/analiz/crm_subat.db"
)

db = SQLDatabase(engine)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tool_list = toolkit.get_tools() + [
    create_chart,
]
print(tool_list)
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
print(prompt_template.input_variables)
system_message = prompt_template.format(dialect="SQLite", top_k=5)
system_message = (
    system_message
    + "\nYou may use create_chart to create a chart. The function signature is:\ncreate_chart(width: int=300, height: int=500, chart_type: str='bar', datasets_label: str='', labels: list=None, data: list=None) -> str\n"
)
print(system_message)
checkpointer = MemorySaver()
agent_executor = create_react_agent(
    llm, tool_list, prompt=system_message, checkpointer=checkpointer
)
