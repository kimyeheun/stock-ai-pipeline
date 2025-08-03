from fastapi import FastAPI
from api.routers import stock

app = FastAPI()
app.include_router(stock.router)
