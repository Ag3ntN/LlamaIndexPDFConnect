from llama_index import download_loader
from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex
from langchain import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
os.listdir(".") 
# downloading the loader from llamahub
PDFReader = download_loader("PDFReader")

loader = PDFReader()
documents = loader.load_data(file=Path('yourfile.pdf'))

# defines the model
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
QA_PROMPT_TMPL = (
    "You have the information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Using the information, please answer next question.: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# structures the data
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
engine = index.as_query_engine(text_qa_template=QA_PROMPT)
questionAsked = input('Please enter your question: ')
response = engine.query(questionAsked)
print(response.response)