import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TextStreamer, pipeline

import os
from constants import *
from utils import *


class PdfQA:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.tokenizer = None
        self.llm = None
        self.qa = None
        self.retriever = None

    @classmethod
    def create_embeddings(cls):
        return HuggingFaceInstructEmbeddings(
            model_name=embedding_model_name, 
            model_kwargs={"device": DEVICE})
    
    @classmethod
    def create_llama_2_model(cls):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            revision="gptq-4bit-128g-actorder_True",
            model_basename=model_basename,
            use_safetensors=True,
            inject_fused_attention=False,
            device=DEVICE,
            quantize_config=None)
        return tokenizer, model
    
    def init_embeddings(self) -> None:
        if self.embedding is None:
            self.embedding = PdfQA.create_embeddings()

    def init_models(self) -> None:
        if self.tokenizer is None and self.llm is None:
            self.tokenizer, self.llm = self.create_llama_2_model()

    def vector_db_pdf(self) -> None:
        pdf_path = self.config.get("pdf_path", None)
        persist_directory = self.config.get("persist_directory", None)
        if persist_directory and os.path.exists(pdf_path):
            self.vectordb = Chroma(persist_directory=persist_directory, 
                                   embedding_function=self.embedding)
        elif pdf_path and os.path.exists(pdf_path):
            # 1. Extract the documents
            loader = PyPDFDirectoryLoader(pdf_path)
            docs = loader.load()
            # 2. Split the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
            texts = text_splitter.split_documents(docs)
            # Create embeddings and add to vector datastore
            self.vectordb = Chroma.from_documents(documents=texts, 
                                                  embedding=self.embedding, 
                                                  persist_directory=persist_directory)
        else:
            raise ValueError("NO PDF FOUND!!!")
        
    def retrieval_qa_chain(self):
        streamer = TextStreamer(self.tokenizer, 
                                skip_prompt=True, 
                                skip_special_tokens=True)
        # 2. Create Text Pipeline
        text_pipeline = pipeline("text-generation", 
                                 model=self.llm, 
                                 tokenizer=self.tokenizer, 
                                 max_new_tokens=1024, 
                                 temperature=0,
                                 top_p=0.95, 
                                 repetition_penalty=1.15,
                                 streamer=streamer)
        # 3. Create HF Pipeline
        llm_pipeline = HuggingFacePipeline(pipeline=text_pipeline, 
                                           model_kwargs={"temperature": 0})
        # 4. Create Template
        template = generate_prompt(
        """
        {context}

        Question: {question}
        """,
        system_prompt=SYSTEM_PROMPT
    )
        # 5. Create Prompt Template
        prompt = PromptTemplate(template=template, input_variables={"context", "question"})
        # 6. Create QA Chain
        self.qa = RetrievalQA.from_chain_type(llm=llm_pipeline,
                                               chain_type="stuff",
                                               retriever=self.vectordb.as_retriever(search_kwargs={"k": 2}),
                                               return_source_documents=True,
                                               chain_type_kwargs={"prompt": prompt},
                                               verbose=True)
        
    def answer_query(self, question:str) -> str:

        answer_dict = self.qa({"query": question})
        answer = answer_dict['result']
        return answer

