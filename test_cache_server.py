from dataclasses import dataclass
from typing import Callable

# 测试
import torch
def fun(x, y):
    print("x=", x, "y=", y)

@dataclass
class test:
    a: int = 1
    b: str = "2"
    c: torch.dtype = torch.float32
    d: Callable = torch.exp
    e: Callable = fun

value = [1, 2, 3, 1.5, "2.5", torch.tensor(3.5, dtype=torch.bfloat16), torch.exp, fun]

import pickle
import hashlib
test_config = test()
print(test_config.__dict__)
test_config.d(torch.tensor(1))
hash_key = hashlib.md5(pickle.dumps(test_config)).hexdigest()
import httpx
response = httpx.post("http://localhost:8000/?hash_key="+hash_key, data=pickle.dumps(value), headers={"Content-Type": "application/octet-stream"})
print(response.text)

response = httpx.get("http://localhost:8000/?hash_key="+hash_key)
if response.content == '':
    rep = None
else:
    rep = pickle.loads(response.content)
print(rep)

response = httpx.get("http://localhost:8000/stats")
print(response.text)

