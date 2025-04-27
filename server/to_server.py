from fastapi import FastAPI
import time  # 模拟处理耗时

app = FastAPI()


@app.get("/chat")
def chat_api(message: str):
    # 模拟AI处理（同步阻塞）
    time.sleep(3)  # 假设处理耗时2秒

    # 一次性返回完整结果
    return {
        "content": "接收到：" + message,
        "reasoning": f"""
        what: 什么是{message}
        why: 为什么要{message}
        how: 怎样{message}
        """
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)