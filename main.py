import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rlcube.models.search import astar_solve

app = FastAPI()


class StateArgs(BaseModel):
    state: list[list[int]]


@app.post("/solve")
def solve(body: StateArgs):
    state = np.array(body.state)
    if state.shape != (6, 4):
        raise HTTPException(status_code=400, detail="state must be a 6x4 matrix")
    if state.min() < 0 or not np.array_equal(
        np.bincount(state.reshape(-1), minlength=6), [4] * 6
    ):
        raise HTTPException(
            status_code=400, detail="state must contain each color 0-5 exactly 4 times"
        )

    steps, _ = astar_solve(state.astype(np.int8))
    if steps is None:
        raise HTTPException(status_code=422, detail="Unable to solve the cube")
    return {"steps": [int(a) for a in steps]}
