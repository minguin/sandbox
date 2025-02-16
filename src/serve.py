#!/usr/bin/env python
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes


def save_data(a: np.numpy, b: pd.DataFrame) -> str:
    """_summary_

    Args:
        a (np.numpy): _description_
        b (pd.DataFrame): _description_

    Returns:
        str: _description_
    """


load_dotenv(verbose=True)
# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
# 2. Create model
model = ChatOpenAI()
# 3. Create parser
parser = StrOutputParser()
# 4. Create chain
chain = prompt_template | model | parser
# 4. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
