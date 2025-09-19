import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

load_dotenv()

DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://myuser:mypassword@127.0.0.1:5432/ragdb')
ASYNC_DB_URL = DB_URL.replace('postgresql+psycopg2', 'postgresql+asyncpg')

engine = create_engine(
    DB_URL
)
async_engine = create_async_engine(
    ASYNC_DB_URL
)
SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)
AsyncSessionLocal = async_sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = async_engine
)

class Base(DeclarativeBase):
    pass

