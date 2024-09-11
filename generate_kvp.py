import logging
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, BaseOutputParser
from io import StringIO
import pandas as pd
import streamlit as st


class CSVStringToDataFrameParser(BaseOutputParser):
    def __init__(self):
        pass

    def parse(self, output: str) -> pd.DataFrame:
        # Strip leading/trailing whitespace and newlines
        output = output.strip()

        # Use StringIO to treat the string as a file-like object
        csv_data = StringIO(output)

        # Read the CSV data into a DataFrame
        df = pd.read_csv(csv_data)

        return df

@st.cache_resource(show_spinner=False)
def loadChain() -> str:
    """
    Loads Chain to Generate KVP from rw extracted Text.

    Args:
        None

    Returns:
        llm_chain: llm_chain to Generate KVP

    Raises:
        Exception: If there's an error during pipeline execution.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading llm_groq_mixtral...")
    try:
        # Define the prompt template
        prompt_template = PromptTemplate(
            input_variables=['extracted_text'],
            template="""
                You are given a block of text extracted from an image. Your task is to structure the information into key-value pairs.
                Example output:
                Key,Value
                
                Output:
                Generate the CSV format output for the given input extracted text: {extracted_text}
                """
                )
        # Initialize the ChatGroq model
        llm_groq_mixtral = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key='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'
        )

        # Create the LLM chain
        llm_chain = prompt_template | llm_groq_mixtral | CSVStringToDataFrameParser()
        logger.info("Loaded llm_groq_mixtral successfully!")
        return llm_chain
    except Exception as e:
        logger.error("Error generating KVP:", exc_info=True)
        raise Exception("Error generating KVP") from e


def get_kvp(extracted_text, llm_chain):
    """
    Fetches medical information for a given entity using the LLM chain.

    Parameters:
        extracted_text : Extracted Text

    Returns:
        df:A dataframe.
               Returns (None, None) if an error occurs.
    """
    try:
        # Run the LLM chain with the provided input data
        output = llm_chain.invoke(extracted_text)

        return output

    except Exception as e:
        # Handle any errors that occur during the process
        print(f"An error occurred in get_kvp: {e}")
        return None

