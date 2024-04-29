#!/usr/bin/env python
# coding: utf-8

import gradio as gr

import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain.schema import Document
from langchain.vectorstores import Chroma, FAISS
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
import time
import re
import os

from vllm_model import ChatLLM
from vllm_wrapper import vLLMWrapper
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess


def get_qa_chain(llm, vector_store, prompt_template):
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)


def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt = cnt + 1
        if (cnt > 6):
            break
        if (len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if (len(bm25_ans + doc.page_content) > max_length):
            break
        bm25_ans = bm25_ans + doc.page_content
        if (cnt > 6):
            break

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                2: {bm25_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, bm25_ans=bm25_ans, question=query)
    return prompt_template


def get_rerank(emb_ans, query):
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, question=query)
    return prompt_template


def question(text, llm, vector_store, prompt_template):
    chain = get_qa_chain(llm, vector_store, prompt_template)

    response = chain({"query": text})
    return response


def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    # docs_sort = sorted(rerank_ans, key = lambda x:x.metadata["id"])
    emb_ans = ""
    for doc in rerank_ans:
        if (len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans


if __name__ == "__main__":

    if os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true":
        print("man, what can i say! -------------------------------------------")

    start = time.time()
    # base = "/app"
    # qwen7 = "/tcdata/qwen/Qwen-7B-Chat"

    base = "/mnt/workspace/Tianchi-LLM-QA"
    qwen7 = base + "/pre_train_model/Qwen-7B-Chat/qwen/Qwen-7B-Chat"
    m3e =  base + "/pre_train_model/M3E-large/Jerry0/M3E-large"
    bge_reranker_large = base + "/pre_train_model/bge-reranker-large/Xorbits/bge-reranker-large"

    # data
    # dp =  DataProcess(pdf_path = "/tcdata/trainning_data.pdf")
    dp = DataProcess(pdf_path=base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    print(len(dp.data))
    data = dp.data
    print("data load ok")

    # Faiss
    faissretriever = FaissRetriever(m3e, data)
    vector_store = faissretriever.vector_store
    print("faissretriever load ok")

    # BM2.5
    bm25 = BM25(data)
    print("bm25 load ok")

    # LLM
    # llm = vLLMWrapper(qwen7)
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # reRank
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")


    def ask_question(question):
        query = question
        max_length = 4000
        # faiss
        faiss_context = faissretriever.GetTopK(query, 15)
        faiss_min_score = 0.0
        if (len(faiss_context) > 0):
            faiss_min_score = faiss_context[0][1]
        cnt = 0
        emb_ans = ""
        for doc, score in faiss_context:
            cnt = cnt + 1
            if (len(emb_ans + doc.page_content) > max_length):
                break
            emb_ans = emb_ans + doc.page_content
            if (cnt > 6):
                break

        # bm2.5
        bm25_context = bm25.GetBM25TopK(query, 15)
        bm25_ans = ""
        cnt = 0
        for doc in bm25_context:
            cnt = cnt + 1
            if (len(bm25_ans + doc.page_content) > max_length):
                break
            bm25_ans = bm25_ans + doc.page_content
            if (cnt > 6):
                break

        emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)
        bm25_inputs = get_rerank(bm25_ans, query)
        emb_inputs = get_rerank(emb_ans, query)

        # rerank emb recall
        rerank_ans = reRank(rerank, 6, query, bm25_context, faiss_context)
        rerank_inputs = get_rerank(rerank_ans, query)

        batch_input = []
        batch_input.append(emb_bm25_merge_inputs)
        batch_input.append(bm25_inputs)
        batch_input.append(emb_inputs)
        batch_input.append(rerank_inputs)
        batch_output = llm.infer(batch_input)
        return batch_output[0].strip(), \
               batch_output[1].strip(), \
               batch_output[2].strip(), \
               batch_output[3].strip()


    demo = gr.Interface(fn=ask_question, inputs="text",
                        outputs=[gr.Textbox(label="answer1"),
                                 gr.Textbox(label="answer2"),
                                 gr.Textbox(label="answer3"),
                                 gr.Textbox(label="answer4")],
                        allow_flagging="never")
    demo.launch(server_name="0.0.0.0")
