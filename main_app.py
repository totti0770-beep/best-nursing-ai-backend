from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import Cache, CORSConfig
from fastapi_cache.backends.inmemory import InMemoryBackend
from pydantic import BaseModel
import httpx
import logging
import uvicorn
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize cache
cache = Cache(backend=InMemoryBackend())

# Constants
JOKE_API_URL = 'https://v2.jokeapi.dev/joke/'
RATE_LIMIT = 1  # Limit to 1 call per second

# Models
class Joke(BaseModel):
    setup: str = None
    delivery: str = None
    joke: str = None
    type: str = None

# Endpoint to get a random joke
@app.get('/joke/random', response_model=Joke)
@sleep_and_retry
@limits(calls=RATE_LIMIT, period=1)
async def get_random_joke():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(JOKE_API_URL + 'Any')
            response.raise_for_status()
            joke_data = response.json()
            if joke_data['type'] == 'single':
                return Joke(joke=joke_data['joke'])
            else:
                return Joke(setup=joke_data['setup'], delivery=joke_data['delivery'])
    except Exception as e:
        logger.error(f'Error fetching joke: {e}')
        raise HTTPException(status_code=500, detail='Error fetching joke')

# Endpoint for dad jokes
@app.get('/joke/dadjoke', response_model=Joke)
@sleep_and_retry
@limits(calls=RATE_LIMIT, period=1)
async def get_dad_joke():
    return await get_joke_by_type('Programming')

# Generic joke endpoint by type
async def get_joke_by_type(joke_type: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(JOKE_API_URL + joke_type)
            response.raise_for_status()
            joke_data = response.json()
            if joke_data['type'] == 'single':
                return Joke(joke=joke_data['joke'])
            else:
                return Joke(setup=joke_data['setup'], delivery=joke_data['delivery'])
    except Exception as e:
        logger.error(f'Error fetching joke by type: {e}')
        raise HTTPException(status_code=500, detail='Error fetching joke')

# Health check
@app.get('/health')
async def health_check():
    return JSONResponse(content={'status': 'OK'})

# Root endpoint
@app.get('/')
async def root():
    return {'message': 'Welcome to the Joke API!'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
