import logging
import json
from logging import config
import urllib.parse

from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import StructuredTool

from quickchart import QuickChart

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

from typing import List, Optional, Union, Literal, Dict
from pydantic import BaseModel, Field


class Dataset(BaseModel):
    label: Optional[str] = Field(None, description="The label for the dataset.")
    data: List[Union[int, float, dict]] = Field(
        ..., description="The data points for the dataset. Should be in {x, y} format for scatter charts."
    )
    backgroundColor: Optional[Union[str, List[str]]] = Field(
        None, description="Background color(s) for the dataset."
    )
    borderColor: Optional[Union[str, List[str]]] = Field(
        None, description="Border color(s) for the dataset."
    )
    borderWidth: Optional[int] = Field(None, description="Width of the dataset border.")
    fill: Optional[Union[bool, str]] = Field(
        None, description="Whether to fill the dataset area."
    )
    type: Optional[str] = Field(
        None, description="Type of the dataset, e.g., line or bar."
    )
    yAxisID: Optional[str] = Field(
        None, description="ID of the Y-axis to bind this dataset to."
    )
    stack: Optional[str] = Field(None, description="Stack group for stacked charts.")


class Data(BaseModel):
    labels: Optional[List[Union[int, str]]] = Field(None, description="Labels for the X-axis. Not used with scatter charts.")
    datasets: List[Dataset] = Field(..., description="List of datasets to be plotted.")


class TitleOptions(BaseModel):
    display: Optional[bool] = Field(
        True, description="Whether to display the chart title."
    )
    text: Optional[str] = Field(None, description="Text of the chart title.")
    fontSize: Optional[int] = Field(None, description="Font size of the title.")
    fontColor: Optional[str] = Field(None, description="Color of the title text.")
    position: Optional[Literal["top", "left", "bottom", "right"]] = Field(
        None, description="Position of the title."
    )


class LegendOptions(BaseModel):
    display: Optional[bool] = Field(
        True, description="Whether to display the chart legend."
    )
    position: Optional[Literal["top", "left", "bottom", "right"]] = Field(
        "top", description="Position of the legend."
    )


class TooltipOptions(BaseModel):
    enabled: Optional[bool] = Field(True, description="Whether tooltips are enabled.")
    mode: Optional[str] = Field(
        None, description="Mode of the tooltip, e.g., index or dataset."
    )
    intersect: Optional[bool] = Field(
        None, description="Whether tooltips should intersect with items."
    )


class AxisTicks(BaseModel):
    beginAtZero: Optional[bool] = Field(
        None, description="Whether the scale should start at zero."
    )
    min: Optional[float] = Field(None, description="Minimum tick value.")
    max: Optional[float] = Field(None, description="Maximum tick value.")


class AxisScale(BaseModel):
    display: Optional[bool] = Field(True, description="Whether to display the axis.")
    ticks: Optional[AxisTicks] = Field(
        None, description="Tick configuration for the axis."
    )
    scaleLabel: Optional[Dict[str, Union[str, bool]]] = Field(
        None, description="Label configuration for the scale."
    )


class Scales(BaseModel):
    xAxes: Optional[List[AxisScale]] = Field(
        None, description="List of X-axis configurations."
    )
    yAxes: Optional[List[AxisScale]] = Field(
        None, description="List of Y-axis configurations."
    )


class Options(BaseModel):
    responsive: Optional[bool] = Field(
        True, description="Whether the chart should be responsive."
    )
    title: Optional[TitleOptions] = Field(None, description="Chart title options.")
    legend: Optional[LegendOptions] = Field(None, description="Chart legend options.")
    tooltips: Optional[TooltipOptions] = Field(
        None, description="Chart tooltip options."
    )
    scales: Optional[Scales] = Field(None, description="Chart scales configuration.")
    maintainAspectRatio: Optional[bool] = Field(
        None, description="Whether to maintain aspect ratio."
    )


class QuickChartConfig(BaseModel):
    type: str = Field(..., description="Type of the chart, e.g., bar, line.")
    data: Data = Field(..., description="Chart data including labels and datasets.")
    options: Optional[Options] = Field(None, description="Additional chart options.")
    width: Optional[int] = Field(500, description="Width of the chart.")
    height: Optional[int] = Field(300, description="Height of the chart.")
    format: Optional[Literal["png", "jpeg", "webp", "svg"]] = Field(
        "png", description="Format of the chart image."
    )
class CreateChartInput(BaseModel):
    config: QuickChartConfig = Field(..., description="QuickChart configuration.")
    width: Optional[int] = Field(500, description="Width of the chart.")
    height: Optional[int] = Field(300, description="Height of the chart.")
    format: Optional[Literal["png", "jpeg", "webp", "svg"]] = Field(
        "png", description="Format of the chart image."
    )

load_dotenv()


@tool("create_chart", args_schema=QuickChartConfig, return_direct=True)
def create_chart(type: str="bar", data: Data=None, options: Options=None, width: int=500, height: int=300, format: str="png") -> str:
    """Create a chart using QuickChart and return the URL. All parameters should conform to QuickChart API.
    Args:
        type (str): Type of the chart (e.g., 'bar', 'line').
        data (Data): Data for the chart.
        options (Options): Options for the chart.
        width (int): Width of the chart.
        height (int): Height of the chart.
        format (str): Format of the chart image.
    Returns:
        str: URL of the generated chart.
    """
    logging.info(
        f"Creating chart with type={type}, width={width}, height={height}, format={format}, data={data}, options={options}"
    )
    qc = QuickChart()
    # Set the chart width and height
    qc.width = width
    qc.height = height
    # qc.config = config.model_dump(exclude_none=True)
    if data:
        data_dict = data.model_dump(exclude_none=True)
    else:
        data_dict = {}
    if options: 
        options_dict = options.model_dump(exclude_none=True)
    else:
        options_dict = {}
    qc.config = {
        "type": type,
        "data": data_dict,
        "options": options_dict,
    }
    qc.format = format
    return qc.get_short_url()


# @tool
# def create_chart(
#     width: int = 800,
#     height: int = 500,
#     chart_type: str = "bar",
#     datasets_label: str = "",
#     labels: list = None,
#     data: list = None,
# ) -> str:
#     """Create a chart using QuickChart and return the URL. All parameters should conform to QuickChart API.
#     Args:
#         width (int): Width of the chart.
#         height (int): Height of the chart.
#         chart_type (str): Type of the chart (e.g., 'bar', 'line').
#         datasets_label (str): Label for the datasets.
#         labels (list): Labels for the x-axis.
#         data (list): Data for the y-axis.
#     Returns:
#         str: URL of the generated chart.
#     """
#     logging.info(
#         f"Creating chart with width={width}, height={height}, chart_type={chart_type}, datasets_label={datasets_label}, labels={labels}, data={data}"
#     )
#     qc = QuickChart()
#     # Set the chart width and height
#     qc.width = width
#     qc.height = height
#     # If chart_type is 'scatter', transform labels and data into a list of {x, y} objects
#     if chart_type == "scatter" and labels and data:
#         data = [{"x": x, "y": y} for x, y in zip(labels, data)]
#         qc.config = {
#             "type": chart_type,
#             "data": {
#                 "datasets": [{"label": datasets_label, "data": data, "fill": False}],
#             },
#         }
#     else:
#         qc.config = {
#             "type": chart_type,
#             "data": {
#                 "labels": labels,
#                 "datasets": [{"label": datasets_label, "data": data, "fill": False}],
#             },
#         }
#     return qc.get_short_url()


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
tool_list = toolkit.get_tools() + [create_chart]
print(tool_list)
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
assert len(prompt_template.messages) == 1
print(prompt_template.input_variables)
system_message = prompt_template.format(dialect="SQLite", top_k=5)
system_message = (
    system_message
    + "\nYou may use create_chart to create a chart." # The function signature is:\ncreate_chart(width: int=300, height: int=500, chart_type: str='bar', datasets_label: str='', labels: list=None, data: list=None) -> str\n"
)
print(system_message)
checkpointer = MemorySaver()
agent_executor = create_react_agent(
    llm, tool_list, prompt=system_message, checkpointer=checkpointer #, debug=True
)
