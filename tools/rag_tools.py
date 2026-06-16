from langchain.tools import tool
from datetime import datetime
from Search.search import RagSearch
import math

allowed = {
    "sqrt": math.sqrt,
    "pow": pow,
    "abs": abs
}



@tool
def check_current_date():
    """
    Returns today's date.
    """
    return str(datetime.now().date())


@tool
def calculator(expression:str):
    """
    Evaluate a mathematical expression.
    """
    return str(eval(expression, {"__builtins__": {}},
        allowed))






