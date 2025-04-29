import os
import logging
import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
# Import Security, Header, Depends
from fastapi import FastAPI, HTTPException, Request, status, Response, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
# Import APIConnectionError
from openai import OpenAI, APIError, AuthenticationError, RateLimitError, BadRequestError, AsyncOpenAI, APIConnectionError
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger
# Optional: Import CORS middleware
# from fastapi.middleware.cors import CORSMiddleware


# --- Logging Configuration ---
# Configure structured JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logHandler.setFormatter(formatter)

logger = logging.getLogger("OpenAI_API")
logger.addHandler(logHandler)
logger.setLevel(logging.INFO) # Adjust level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as needed

# --- Configuration ---
class Settings(BaseSettings):
    """Loads configuration from environment variables."""
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    # Add API Key for securing *this* API
    # Load this from environment variables in Vercel
    api_access_key: str = Field(..., env="API_ACCESS_KEY")
    # Optional: Allowed origins for CORS
    # allowed_origins: List[str] = ["http://localhost:3000", "https://your-frontend-domain.com"]


    class Config:
        load_dotenv()
        env_file = '.env'
        extra = 'ignore'

settings = Settings()

# --- API Key Security ---
API_KEY_NAME = "X-API-KEY" # Define header name
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) # Set auto_error=False for custom handling if needed, True is simpler

async def get_api_key(api_key_header: Optional[str] = Security(api_key_header)):
    """
    Dependency function to validate the API key provided in the X-API-KEY header.

    Raises:
        HTTPException: 401 Unauthorized if the key is missing or invalid.

    Returns:
        str: The validated API key.
    """
    if api_key_header is None:
        logger.warning("Missing API Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in header",
        )
    if api_key_header == settings.api_access_key:
        return api_key_header
    else:
        logger.warning(f"Invalid API Key received: '{api_key_header[:4]}...'") # Log only prefix
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# --- OpenAI Client Initialization ---
client: Optional[AsyncOpenAI] = None # Explicitly type hint client
if not settings.openai_api_key or settings.openai_api_key == "...": # Check if key is missing or placeholder
    logger.critical("OPENAI_API_KEY is not set or is a placeholder in environment/config. OpenAI client cannot be initialized.")
else:
    try:
        # Use AsyncOpenAI for async operations, especially streaming
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAI AsyncClient initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing OpenAI client: {e}", exc_info=True)
        client = None # Ensure client is None on error

# --- FastAPI Application ---
app = FastAPI(
    title="OpenAI Interaction API",
    description="API to interact with OpenAI's Chat Completions, supporting streaming.",
    version="1.0.0"
    # Note: Dependencies can be added globally here or per-route below
)

# Optional: Add CORS middleware if frontend is on a different domain
# Make sure API_KEY_NAME is in allow_headers if using CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.allowed_origins,
#     allow_credentials=True,
#     allow_methods=["*"], # Or restrict to ["GET", "POST"]
#     allow_headers=["*", API_KEY_NAME], # Ensure custom header is allowed
# )


# --- Middleware ---
@app.middleware("http")
async def add_process_time_header_and_log(request: Request, call_next):
    start_time = time.time()
    # Add basic security header
    response = Response(status_code=500) # Default response if error before response generation
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Content-Type-Options"] = "nosniff" # Basic security header
        log_extra = {
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host,
            "client_port": request.client.port,
            "status_code": response.status_code,
            "process_time": f"{process_time:.4f}"
        }
        logger.info("Request processed", extra=log_extra)
    except Exception as e:
        process_time = time.time() - start_time
        log_extra = {
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host,
            "client_port": request.client.port,
            "status_code": 500, # Assuming internal server error if exception reaches here
            "process_time": f"{process_time:.4f}"
        }
        logger.error(f"Unhandled exception during request processing: {e}", exc_info=True, extra=log_extra)
        # Return a generic error response if one wasn't already generated
        if not isinstance(response, JSONResponse):
             response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal Server Error"},
            )
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Content-Type-Options"] = "nosniff"

    return response


# --- Request Models ---
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

class OpenAIRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo", description="ID of the model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')")
    prompt: Optional[str] = Field(None, description="The main user prompt (alternative to messages[-1].content)")
    messages: Optional[List[Message]] = Field(None, description="A list of messages describing the conversation history")
    system_prompt: Optional[str] = Field(None, description="An optional system message to guide the model's behavior")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature (0-2). Higher values make output more random.")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum number of tokens to generate (default depends on model).") # Changed default
    # Add other OpenAI parameters as needed (e.g., top_p, frequency_penalty, presence_penalty)
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress.")
    # Example: Adding top_p
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter.")

    # Validation: Ensure either prompt or messages are provided
    # Pydantic v2 validator coming soon, for now handle in endpoint logic

# --- Custom Exception Handlers ---
@app.exception_handler(APIError)
async def openai_api_exception_handler(request: Request, exc: APIError):
    logger.error(f"OpenAI API Error: Status={exc.status_code}, Message={exc.message}", exc_info=True, extra={"request_body": exc.body, "request_params": exc.request.params})
    return JSONResponse(
        status_code=exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"OpenAI API Error: {exc.message or str(exc)}", "type": "openai_api_error"},
    )

@app.exception_handler(AuthenticationError)
async def openai_auth_exception_handler(request: Request, exc: AuthenticationError):
    logger.error(f"OpenAI Authentication Error: {exc.message}", exc_info=False) # Don't log stack trace for auth errors usually
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": f"OpenAI Authentication Error: {exc.message or 'Invalid API Key'}", "type": "openai_auth_error"},
    )

@app.exception_handler(RateLimitError)
async def openai_rate_limit_exception_handler(request: Request, exc: RateLimitError):
    logger.warning(f"OpenAI Rate Limit Error: {exc.message}", exc_info=False) # Warning level might be appropriate
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"OpenAI Rate Limit Error: {exc.message or 'Rate limit exceeded'}", "type": "openai_rate_limit_error"},
    )

@app.exception_handler(BadRequestError)
async def openai_bad_request_handler(request: Request, exc: BadRequestError):
     logger.warning(f"OpenAI Bad Request Error: {exc.message}", exc_info=True, extra={"request_body": exc.body}) # Log body for debugging bad requests
     return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"OpenAI Bad Request Error: {exc.message or 'Invalid request parameters'}", "type": "openai_bad_request_error", "param": getattr(exc, 'param', None)},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Handle FastAPI's own HTTPExceptions (like validation errors)
    logger.warning(f"HTTP Exception: Status={exc.status_code}, Detail={exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}, # Keep FastAPI's default detail format
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catch-all for unexpected errors
    logger.critical(f"Unhandled exception: {exc}", exc_info=True) # Use critical for truly unexpected errors
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected internal server error occurred.", "type": "internal_server_error"}, # Avoid exposing raw error details
    )


# --- Streaming Helper ---
async def stream_openai_response(openai_stream: AsyncGenerator[Any, None]):
    """Asynchronously yields chunks from the OpenAI stream. Formats as SSE."""
    try:
        async for chunk in openai_stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                 # SSE format: yield f"data: {json.dumps({'content': content})}\n\n"
                 # Ensure data is properly JSON encoded for SSE
                 sse_data = json.dumps({"content": content})
                 yield f"data: {sse_data}\n\n" # Yield in SSE format
    except Exception as e:
        logger.error(f"Error during OpenAI stream processing: {e}", exc_info=True)
        # Optionally yield an error message to the client stream, formatted as SSE comment or error event
        error_data = json.dumps({"error": str(e)})
        # Send an 'error' event according to SSE spec
        yield f"event: error\ndata: {error_data}\n\n"
    finally:
         # Optionally send a 'done' event
         # done_data = json.dumps({"message": "Stream finished"})
         # yield f"event: done\ndata: {done_data}\n\n"
         logger.info("OpenAI stream finished.")


# --- API Endpoints ---
# Secure this specific endpoint using the API key dependency
@app.post(
    "/openai_api",
    summary="Interact with OpenAI Chat Completions API",
    response_description="OpenAI completion result or stream",
    dependencies=[Depends(get_api_key)] # Apply security here
)
async def call_openai_api(request_data: OpenAIRequest):
    """
    Receives parameters and forwards the request to the OpenAI Chat Completions API.

    Requires a valid API key in the X-API-KEY header.
    Handles construction of the message payload, streaming, and potential errors.
    """
    if not client:
         logger.error("Attempted API call but OpenAI client is not initialized.")
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI client is not initialized. Check API key and server logs."
        )

    request_id = os.urandom(8).hex() # Generate a simple request ID for logging
    log_extra = {"request_id": request_id, "model": request_data.model, "stream": request_data.stream}
    logger.info("Received OpenAI API request", extra=log_extra)


    messages: List[Dict[str, str]] = []

    # 1. Add System Prompt (if provided)
    if request_data.system_prompt:
        messages.append({"role": "system", "content": request_data.system_prompt})

    # 2. Add Previous Chat Messages (if provided)
    if request_data.messages:
        has_system_in_messages = any(m.role == "system" for m in request_data.messages)
        if request_data.system_prompt and has_system_in_messages:
             logger.warning("System prompt provided in both 'system_prompt' and 'messages'", extra=log_extra)
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide system prompt either via 'system_prompt' field or within 'messages', but not both."
            )
        messages.extend([msg.model_dump() for msg in request_data.messages])

    # 3. Add Current User Prompt (if provided)
    if request_data.prompt:
        messages.append({"role": "user", "content": request_data.prompt})

    # 4. Validation: Ensure we have at least one non-system message
    if not any(msg["role"] == "user" or msg["role"] == "assistant" for msg in messages):
        logger.warning("Request lacks user/assistant message content", extra=log_extra)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request must include a 'prompt' or at least one 'user' or 'assistant' message in 'messages'."
        )

    # 5. Call OpenAI API
    try:
        logger.debug(f"Calling OpenAI with messages: {messages}", extra=log_extra)
        # Use await with the AsyncOpenAI client
        response_stream = await client.chat.completions.create(
            model=request_data.model,
            messages=messages,
            temperature=request_data.temperature,
            max_tokens=request_data.max_tokens,
            top_p=request_data.top_p,
            stream=request_data.stream,
            # Add other parameters here if needed
        )

        if request_data.stream:
            logger.info("Streaming response initiated.", extra=log_extra)
            # Return a StreamingResponse, using the async generator
            # Use 'text/event-stream' for Server-Sent Events
            # Changed media_type to text/event-stream
            return StreamingResponse(stream_openai_response(response_stream), media_type="text/event-stream")
        else:
            # For non-streaming, the response object is already available
            logger.info("Non-streaming response received successfully.", extra=log_extra)
            # The response object from openai v1.x is directly serializable by FastAPI
            return response_stream

    except (APIError, AuthenticationError, RateLimitError, BadRequestError) as e:
        # Logged by specific handlers, just re-raise
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the API call
        logger.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True, extra=log_extra)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while communicating with OpenAI." # Avoid leaking details
        ) from e


# Enhanced Health Check
@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check", tags=["Management"])
async def health_check():
    """
    Checks API health and connectivity to OpenAI.
    Returns 200 OK only if the client is initialized and OpenAI API is reachable.
    Otherwise, returns 503 Service Unavailable with details.
    Does not require API key authentication.
    """
    logger.debug("Health check requested.")
    openai_status = "unknown" # Default status
    openai_reachable = False
    # Default detail dictionary, assuming error until proven otherwise
    status_detail = {"status": "error", "openai_status": openai_status, "openai_reachable": openai_reachable}

    if not client:
        openai_status = "not_initialized"
        logger.warning("Health check: OpenAI client not initialized.")
        status_detail.update({"openai_status": openai_status, "openai_reachable": False})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=status_detail
        )
    else:
        # Try a lightweight API call to check connectivity/authentication
        try:
            logger.debug("Health check: Attempting to list OpenAI models...")
            # Listing models is relatively inexpensive
            await client.models.list(limit=1)
            openai_reachable = True
            openai_status = "ok" # Status is 'ok' only if API call succeeds
            logger.debug("Health check: OpenAI API reached successfully.")
            # Update detail for success case
            status_detail = {"status": "ok", "openai_status": openai_status, "openai_reachable": openai_reachable}
            return status_detail # Return 200 OK

        except AuthenticationError:
            openai_status = "auth_error"
            logger.error("Health check: OpenAI authentication error. Check OPENAI_API_KEY.")
        except APIConnectionError as e:
            openai_status = "connection_error"
            logger.error(f"Health check: OpenAI connection error: {e}")
        except APIError as e: # Catch other specific OpenAI API errors
            openai_status = f"api_error ({e.status_code})"
            logger.error(f"Health check: OpenAI API error: Status={e.status_code}, Message={e.message}")
        except Exception as e: # Catch any other unexpected errors during the check
            openai_status = "check_failed"
            logger.error(f"Health check: Unexpected error during OpenAI check: {e}", exc_info=True)

        # If any exception occurred during the try block, update detail and raise 503
        status_detail.update({"openai_status": openai_status, "openai_reachable": False})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=status_detail
        )

# The 'app' variable is automatically detected by Vercel's Python runtime.

