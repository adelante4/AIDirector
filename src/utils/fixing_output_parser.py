import logging
import string

from llama_index.core.output_parsers import ChainableOutputParser, PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram

from src.director.prompts import output_fixing_template


class FixingOutputParser(ChainableOutputParser):
    def __init__(self, output_class):
        super().__init__()
        self.output_class = output_class
        self.pydantic_parser = PydanticOutputParser(output_class)

    def parse(self, llm_output: str):
        try:
            return self.pydantic_parser.parse(llm_output)

        except Exception as e:
            logging.info(f"Error parsing output: {e}")
            fixed_output = self.fix_output(llm_output, str(e))
            return fixed_output

    def fix_output(self, text: str, error: str):
        logging.info(f"Fixing output: {text}")
        try:
            text = ''.join(filter(lambda x: x in string.printable, text))
            return self.pydantic_parser.parse(text)
        except Exception as e:
            logging.info("using fixing parser: \nError: {}".format(e))
            return LLMTextCompletionProgram.from_defaults(
                output_cls=self.output_class,
                prompt_template_str=output_fixing_template.get_template(),
            )(completion=text, error=error, instructions=self.pydantic_parser.get_format_string())

    def get_format_string(self):
        return self.pydantic_parser.get_format_string()
