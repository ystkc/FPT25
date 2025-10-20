import os
import fastapi
import hashlib
import uvicorn

app = fastapi.FastAPI()

# 纯内存缓存
# 3种功能：
# 1. 客户端提交缓存
# 2. 客户端查询缓存
# 3. 查看当前缓存情况

cache = {}
USE_DISK_CACHE = True
save_path = 'cache'
os.chdir(os.path.dirname(__file__))
print(os.path.dirname(__file__))
os.mkdir(save_path) if not os.path.exists(save_path) else None

@app.get("/")
async def get_cache(_: fastapi.Request, hash_key: str = None):
    c = cache.get(hash_key, b"")
    if not c and USE_DISK_CACHE:
        # 尝试从硬盘中读取
        try:
            with open(f"{save_path}/{hash_key}", "rb") as f:
                c = f.read()
            cache[hash_key] = c
        except:
            pass
    return fastapi.Response(content=c, media_type="application/octet-stream")

@app.post("/")
async def submit_cache(request: fastapi.Request, hash_key: str):
    value = await request.body()
    cache[hash_key] = value
    if USE_DISK_CACHE:
        with open(f"{save_path}/{hash_key}", "wb") as f:
            f.write(value)
        return {"message": f"Cache submitted and saved to disk ({len(value)}bytes)"}
    return {"message": f"Cache submitted successfully ({len(value)}bytes)"}

@app.get("/stats")
async def get_stats(_: fastapi.Request):
    return {"cache_count": len(cache), "cache_size": sum(len(v) for v in cache.values())}

uvicorn.run(app, host="0.0.0.0", port=8000)
