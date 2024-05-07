import hashlib
import logging
import uuid

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from tinydb import TinyDB, Query

from bm25_retriever import BM25
from faiss_retriever import FaissRetriever
from pdf_parse import DataProcess
from rerank_model import reRankLLM
from run import get_emb_bm25_merge, get_rerank, reRank
from vllm_model import ChatLLM


class Result:
    """
    {
        "success": bool,
        "message": str,
        "data": object
    }
    """

    @staticmethod
    def success(data):
        return {"success": True, "message": "success", "data": data}

    @staticmethod
    def error(message):
        return {"success": False, "message": message, "data": None}


logging.basicConfig(level=logging.INFO)

app = FastAPI()
db = TinyDB('./db.json')
db_users = db.table('users')
db_pdfs = db.table('pdfs')

base = "/mnt/workspace/Tianchi-LLM-QA"
qwen7 = base + "/pre_train_model/Qwen-7B-Chat/qwen/Qwen-7B-Chat"
m3e = base + "/pre_train_model/M3E-large/Jerry0/M3E-large"
bge_reranker_large = base + "/pre_train_model/bge-reranker-large/Xorbits/bge-reranker-large"

# LLM
llm = ChatLLM(qwen7)
print("llm qwen load ok")

# reRank
rerank = reRankLLM(bge_reranker_large)
print("rerank model load ok")


def calc_vector_store(pdf_id, file_path):
    logging.info(f"calc_vector_store: load data {pdf_id}")
    dp = DataProcess(pdf_path=file_path)
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
    logging.info(f"calc_vector_store: data load ok {pdf_id}")

    logging.info(f"calc_vector_store: FaissRetriever {pdf_id}")
    faissretriever = FaissRetriever(m3e, data)
    faissretriever.SaveLocalData(f"./vec_store/{pdf_id}")
    print(f"calc_vector_store: Faissretriever load ok {pdf_id}")

    logging.info(f"calc_vector_store: BM25 {pdf_id}")
    bm25 = BM25(data)
    bm25.SaveLocalData(f"./bm/{pdf_id}")
    print(f"calc_vector_store: BM25 load ok {pdf_id}")

    db_pdfs.update({"vector_store": True}, Query().id == pdf_id)


@app.post("/upload_pdf")
async def create_upload_file(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    # get md5 hash of file
    md5 = hashlib.md5(pdf.file.read()).hexdigest()
    if db_pdfs.get(Query().md5 == md5) is not None:
        return Result.error("file already exists")

    # save file to disk
    pdf_id = str(uuid.uuid4())
    file_path = f"./pdfs/{pdf_id}.pdf"

    try:
        with open(file_path, "wb") as f:
            f.write(pdf.file.read())
    except Exception as e:
        return Result.error(str(e))

    # save file to db
    db_pdfs.insert({"id": pdf_id, "md5": md5, "path": file_path, "vector_store": False})
    background_tasks.add_task(calc_vector_store, pdf_id, file_path)

    return Result.success({"id": pdf_id})


@app.get("/pdfs")
async def get_pdfs():
    return db_pdfs.all()


@app.get("/chat")
async def query_pdf(pdf_id: str, question: str):
    faissretriever = FaissRetriever()
    faissretriever.LoadLocalData(m3e, f"./vec_store/{pdf_id}")
    print("faissretriever load ok")

    # BM2.5
    bm25 = BM25()
    bm25.LoadLocalData(f"./bm/{pdf_id}")
    print("bm25 load ok")

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

    res = {}
    res["answer_1"] = batch_output[0]
    res["answer_2"] = batch_output[1]
    res["answer_3"] = batch_output[2]
    res["answer_4"] = batch_output[3]
    res["answer_5"] = emb_ans
    res["answer_6"] = bm25_ans
    res["answer_7"] = rerank_ans
    if (faiss_min_score > 500):
        res["answer_5"] = "无答案"
    else:
        res["answer_5"] = str(faiss_min_score)

    return Result.success(res)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
