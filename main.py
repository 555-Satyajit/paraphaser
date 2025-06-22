import os
import time
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import json
from dotenv import load_dotenv
import random
import traceback
import redis

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    
# Redis configuration for persistent rate limiting
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Pydantic models
class ParaphraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    num_variations: int = Field(default=3, ge=1, le=5)

class ParaphraseResponse(BaseModel):
    original_text: str
    paraphrases: List[str]
    total_variations: int
    processing_time: float
    service_used: str
    tokens_used: Optional[int] = None
    user_requests_remaining: Optional[int] = None
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None

class RequestQueue:
    """Queue system for handling rate-limited requests"""
    
    def __init__(self):
        self.queue = deque()
        self.processing = False
        self.last_request_time = 0
        self.min_interval = 5.5  # 5.5 seconds between requests (safer than 5s)
        logger.info("RequestQueue initialized")
    
    async def add_request(self, request_data, user_id):
        """Add request to queue and return the result"""
        queue_item = {
            'request_data': request_data,
            'user_id': user_id,
            'timestamp': time.time(),
            'future': asyncio.Future()
        }
        self.queue.append(queue_item)
        logger.info(f"Request added to queue for user {user_id}")
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self.process_queue())
        
        # Wait for the future to complete and return the result
        return await queue_item['future']
    
    async def process_queue(self):
        """Process queued requests with rate limiting"""
        if self.processing:
            return
        
        self.processing = True
        logger.info("Started processing queue")
        
        try:
            while self.queue:
                item = self.queue.popleft()
                
                # Wait for rate limit interval
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                
                if time_since_last < self.min_interval:
                    wait_time = self.min_interval - time_since_last
                    logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                
                try:
                    # Process the actual request
                    result = await processor.paraphrase_text(
                        item['request_data'].text,
                        item['request_data'].num_variations
                    )
                    
                    # Add queue info to the result dictionary
                    result['queue_position'] = 0
                    result['estimated_wait_time'] = 0.0
                    
                    item['future'].set_result(result)
                    logger.info(f"Request processed successfully for user {item['user_id']}")
                    
                except Exception as e:
                    logger.error(f"Error processing request for user {item['user_id']}: {e}")
                    item['future'].set_exception(e)
                
                self.last_request_time = time.time()
                
        finally:
            self.processing = False
            logger.info("Queue processing completed")
    
    def get_queue_status(self):
        """Get current queue status"""
        return {
            'queue_length': len(self.queue),
            'estimated_wait_time': len(self.queue) * self.min_interval,
            'processing': self.processing
        }

class UserRateLimiter:
    def __init__(self):
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            # Fallback to in-memory storage
            self.redis_client = None
            self.user_requests = defaultdict(lambda: {"count": 0, "date": None})
            logger.warning("Falling back to in-memory rate limiting")
        
        self.daily_limit_per_user = 10
        self.failed_requests = defaultdict(int)
        self.cleanup_interval = 3600
        self.last_cleanup = time.time()
        logger.info(f"UserRateLimiter initialized with daily limit: {self.daily_limit_per_user}")
    
    def cleanup_old_data(self):
        """Clean up old tracking data"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            # Remove old failed request counts
            self.failed_requests.clear()
            self.last_cleanup = current_time
            logger.info("Cleaned up old rate limiter data")

    def get_user_id(self, request: Request) -> str:
        """Generate user ID from IP address and User-Agent for better uniqueness"""
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        # Combine IP and partial user agent for better uniqueness
        unique_string = f"{client_ip}:{user_agent[:50]}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def check_user_limit(self, user_id: str) -> bool:
        """Check if user has exceeded daily limit"""
        self.cleanup_old_data()
        
        if self.redis_client:
            return self._check_user_limit_redis(user_id)
        else:
            return self._check_user_limit_memory(user_id)

    def _check_user_limit_redis(self, user_id: str) -> bool:
        """Redis-based rate limiting"""
        today = datetime.now().date().isoformat()
        key = f"user_requests:{user_id}:{today}"
        
        try:
            current_count = self.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= self.daily_limit_per_user:
                logger.warning(f"User {user_id} exceeded daily limit (Redis)")
                return False
            
            # Increment counter and set expiration
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 86400)  # 24 hours
            pipe.execute()
            
            logger.info(f"User {user_id} request count: {current_count + 1}")
            return True
            
        except Exception as e:
            logger.error(f"Redis error for user {user_id}: {e}")
            # Fallback to allowing request
            return True

    def _check_user_limit_memory(self, user_id: str) -> bool:
        """Fallback in-memory rate limiting"""
        today = datetime.now().date()
        user_data = self.user_requests[user_id]
        
        if user_data["date"] != today:
            user_data["count"] = 0
            user_data["date"] = today
        
        if user_data["count"] >= self.daily_limit_per_user:
            logger.warning(f"User {user_id} exceeded daily limit (memory)")
            return False
        
        user_data["count"] += 1
        return True

    def get_user_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user today"""
        if self.redis_client:
            try:
                today = datetime.now().date().isoformat()
                key = f"user_requests:{user_id}:{today}"
                current_count = self.redis_client.get(key)
                current_count = int(current_count) if current_count else 0
                return max(0, self.daily_limit_per_user - current_count)
            except Exception as e:
                logger.error(f"Redis error getting remaining requests: {e}")
                return self.daily_limit_per_user
        else:
            # Fallback to memory
            today = datetime.now().date()
            user_data = self.user_requests[user_id]
            
            if user_data["date"] != today:
                return self.daily_limit_per_user
            
            return max(0, self.daily_limit_per_user - user_data["count"])

    def log_failed_request(self, user_id: str):
        """Log a failed request for the user"""
        self.failed_requests[user_id] += 1
        logger.info(f"Failed request logged for user {user_id}")
    
class GeminiProcessor:
    """Process text using Google Gemini API with improved error handling"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.error("GEMINI_API_KEY environment variable is not set!")
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        logger.info("GEMINI_API_KEY found and loaded")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-1.5-flash"
        self.session = None
        
        # Enhanced rate limiting tracking
        self.requests_count = 0
        self.last_reset = time.time()
        self.daily_requests = 0
        self.daily_reset = time.time()
        
        # Performance tracking
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.success_count = 0
        
        logger.info(f"GeminiProcessor initialized with model: {self.model}")
        
    async def get_session(self):
        if self.session is None:
            try:
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    keepalive_timeout=30
                )
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    connector=connector
                )
                logger.info("HTTP session created successfully")
            except Exception as e:
                logger.error(f"Error creating HTTP session: {e}")
                raise
        return self.session
    
    def check_rate_limits(self):
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset minute counter
        if current_time - self.last_reset >= 60:
            self.requests_count = 0
            self.last_reset = current_time
        
        # Reset daily counter
        if current_time - self.daily_reset >= 86400:
            self.daily_requests = 0
            self.daily_reset = current_time
        
        # More conservative limits for stability
        if self.requests_count >= 10:  # 10/min instead of 12
            logger.warning("Minute rate limit exceeded")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": 60 - (current_time - self.last_reset),
                    "type": "minute_limit"
                }
            )
        
        if self.daily_requests >= 1200:  # 1200/day instead of 1400 for buffer
            logger.warning("Daily rate limit exceeded")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Daily limit exceeded", 
                    "retry_after": 86400 - (current_time - self.daily_reset),
                    "type": "daily_limit"
                }
            )
    
    async def call_gemini_api_with_retry(self, prompt: str, system_instruction: str = None, max_retries: int = 3) -> Dict:
        """Make API call with exponential backoff retry"""
        
        for attempt in range(max_retries):
            try:
                return await self.call_gemini_api(prompt, system_instruction)
            except HTTPException as e:
                if e.status_code == 429:
                    # Don't retry rate limit errors
                    raise
                elif attempt == max_retries - 1:
                    # Last attempt failed
                    raise
                else:
                    # Exponential backoff: 2^attempt + random jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API call failed, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"API call error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                else:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API call failed, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
    
    async def call_gemini_api(self, prompt: str, system_instruction: str = None) -> Dict:
        """Make API call to Gemini"""
        start_time = time.time()
        self.check_rate_limits()
        
        try:
            session = await self.get_session()
            
            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            
            contents = [{"parts": [{"text": prompt}]}]
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 1,
                    "topP": 0.8,
                    "maxOutputTokens": 512,
                    "stopSequences": []
                }
            }
            
            if system_instruction:
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Paraphraser-API/1.0"
            }
            
            logger.info(f"Making API call to Gemini: {url}")
            
            async with session.post(url, json=payload, headers=headers) as response:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                self.requests_count += 1
                self.daily_requests += 1
                
                logger.info(f"API response status: {response.status}, time: {response_time:.3f}s")
                
                if response.status == 200:
                    result = await response.json()
                    
                    if 'candidates' in result and result['candidates']:
                        content = result['candidates'][0]['content']['parts'][0]['text']
                        
                        tokens_used = None
                        if 'usageMetadata' in result:
                            tokens_used = result['usageMetadata'].get('totalTokenCount', 0)
                        
                        self.success_count += 1
                        logger.info("API call successful")
                        
                        return {
                            'content': content.strip(),
                            'tokens_used': tokens_used,
                            'success': True,
                            'response_time': response_time
                        }
                    else:
                        self.error_count += 1
                        logger.error("No content generated from API")
                        return {'success': False, 'error': 'No content generated'}
                
                elif response.status == 429:
                    self.error_count += 1
                    error_text = await response.text()
                    logger.error(f"API rate limit exceeded: {error_text}")
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "API rate limit exceeded",
                            "details": error_text,
                            "retry_after": 60
                        }
                    )
                else:
                    self.error_count += 1
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return {'success': False, 'error': f"API error {response.status}: {error_text}"}
                    
        except HTTPException:
            raise
        except Exception as e:
            self.error_count += 1
            logger.error(f"Gemini API error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    async def paraphrase_text(self, text: str, num_variations: int = 3) -> Dict:
        """Generate paraphrases using Gemini with retry logic"""
        start_time = time.time()
        logger.info(f"Starting paraphrasing for text: {text[:50]}...")
        
        system_instruction = f"""You are a professional paraphrasing assistant. Your task is to:
1. Generate {num_variations} different paraphrases of the given text
2. Maintain the original meaning while varying the structure and word choice
3. Ensure each paraphrase is distinct and natural
4. Return each paraphrase on a separate line, numbered 1, 2, 3, etc.
5. Return ONLY the numbered paraphrases, no additional text or explanations"""
        
        prompt = f"Please generate {num_variations} paraphrases of this text: {text}"
        
        try:
            result = await self.call_gemini_api_with_retry(prompt, system_instruction)
            processing_time = time.time() - start_time
            
            if result['success']:
                content = result['content']
                logger.info(f"Paraphrasing successful, parsing {len(content)} characters")
                
                # Parse the numbered paraphrases
                paraphrases = []
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                        # Remove the number and period
                        paraphrase = line.split('.', 1)[1].strip()
                        if paraphrase and paraphrase.lower() != text.lower():
                            paraphrases.append(paraphrase)
                
                logger.info(f"Generated {len(paraphrases)} paraphrases")
                
                return {
                    'original_text': text,
                    'paraphrases': paraphrases[:num_variations],
                    'total_variations': len(paraphrases),
                    'processing_time': round(processing_time, 3),
                    'service_used': f'Google Gemini {self.model}',
                    'tokens_used': result.get('tokens_used')
                }
            else:
                logger.error(f"Paraphrasing failed: {result.get('error', 'Unknown error')}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Paraphrasing failed: {result.get('error', 'Unknown error')}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Paraphrasing error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail="Internal error during paraphrasing"
            )
    
    def get_performance_stats(self):
        """Get performance statistics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        total_requests = self.success_count + self.error_count
        success_rate = (self.success_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': round(success_rate, 2),
            'avg_response_time': round(avg_response_time, 3),
            'recent_response_times': list(self.response_times)[-10:]  # Last 10
        }
    
    async def close(self):
        if self.session:
            await self.session.close()
            logger.info("HTTP session closed")

# Initialize components with error handling
try:
    processor = GeminiProcessor()
    logger.info("GeminiProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GeminiProcessor: {e}")
    raise

try:
    user_limiter = UserRateLimiter()
    logger.info("UserRateLimiter initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize UserRateLimiter: {e}")
    raise

try:
    request_queue = RequestQueue()
    logger.info("RequestQueue initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RequestQueue: {e}")
    raise

# Dependency functions
async def check_user_rate_limit(request: Request):
    try:
        user_id = user_limiter.get_user_id(request)
        logger.info(f"Checking rate limit for user: {user_id}")
        
        if not user_limiter.check_user_limit(user_id):
            remaining = user_limiter.get_user_remaining_requests(user_id)
            logger.warning(f"User {user_id} exceeded rate limit")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Daily user limit exceeded",
                    "daily_limit": user_limiter.daily_limit_per_user,
                    "remaining_today": remaining,
                    "reset_time": "00:00 UTC tomorrow",
                    "suggestion": "Try again tomorrow or upgrade to premium"
                }
            )
        
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rate limit check: {e}")
        raise HTTPException(status_code=500, detail="Error checking rate limits")

# FastAPI app
app = FastAPI(
    title="Enhanced Gemini Paraphraser",
    description="Production-ready paraphrasing API with queue management and monitoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return {"error": "Internal server error", "detail": str(exc)}

@app.get("/")
async def root():
    """Root endpoint for basic health check"""
    return {
        "message": "Enhanced Gemini Paraphraser API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "paraphrase": "/paraphrase",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    try:
        perf_stats = processor.get_performance_stats()
        queue_status = request_queue.get_queue_status()
        
        return {
            "status": "healthy",
            "service": "Gemini API",
            "model": processor.model,
            "daily_requests_used": processor.daily_requests,
            "minute_requests_used": processor.requests_count,
            "user_daily_limit": user_limiter.daily_limit_per_user,
            "queue_length": queue_status["queue_length"],
            "success_rate": perf_stats["success_rate"],
            "avg_response_time": perf_stats["avg_response_time"],
            "api_key_configured": bool(processor.api_key)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/usage")
async def usage_stats():
    """Get detailed usage statistics"""
    try:
        perf_stats = processor.get_performance_stats()
        queue_status = request_queue.get_queue_status()
        
        return {
            "api_limits": {
                "requests_this_minute": processor.requests_count,
                "requests_today": processor.daily_requests,
                "minute_limit": 10,
                "daily_limit": 1200,
                "minute_remaining": 10 - processor.requests_count,
                "daily_remaining": 1200 - processor.daily_requests
            },
            "user_limits": {
                "daily_limit_per_user": user_limiter.daily_limit_per_user
            },
            "performance": perf_stats,
            "queue": queue_status
        }
    except Exception as e:
        logger.error(f"Usage stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-status")
async def user_status(request: Request):
    """Get user's current rate limit status"""
    try:
        user_id = user_limiter.get_user_id(request)
        return {
            "user_id": user_id,
            "daily_limit": user_limiter.daily_limit_per_user,
            "requests_remaining": user_limiter.get_user_remaining_requests(user_id),
            "failed_requests": user_limiter.failed_requests.get(user_id, 0),
            "reset_time": "00:00 UTC tomorrow"
        }
    except Exception as e:
        logger.error(f"User status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue-status")
async def queue_status():
    """Get current queue status"""
    try:
        return request_queue.get_queue_status()
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase_endpoint(
    request_data: ParaphraseRequest,
    request: Request,
    user_id: str = Depends(check_user_rate_limit)
):
    """Generate paraphrases of the provided text with queue management"""
    try:
        logger.info(f"Paraphrase request from user {user_id}: {request_data.text[:50]}...")
        
        # Add to queue and wait for processing - this now returns the actual result dict
        result = await request_queue.add_request(request_data, user_id)
        
        # Add user info to response (result is now a dict)
        result["user_requests_remaining"] = user_limiter.get_user_remaining_requests(user_id)
        
        logger.info(f"Paraphrase request completed for user {user_id}")
        return ParaphraseResponse(**result)
        
    except Exception as e:
        # Log failed request but don't count against user limit
        user_limiter.log_failed_request(user_id)
        logger.error(f"Request failed for user {user_id}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.get("/metrics")
async def metrics():
    """Get comprehensive metrics for monitoring"""
    try:
        perf_stats = processor.get_performance_stats()
        queue_status = request_queue.get_queue_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "api_usage": {
                "daily_requests": processor.daily_requests,
                "daily_limit": 1200,
                "daily_usage_percent": round((processor.daily_requests / 1200) * 100, 2),
                "minute_requests": processor.requests_count,
                "minute_limit": 10
            },
            "performance": perf_stats,
            "queue": queue_status,
            "system": {
                "uptime": time.time() - processor.daily_reset,
                "total_users_served": len(user_limiter.user_requests) if not user_limiter.redis_client else "N/A (Redis)"
            }
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        await processor.close()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Move startup logging here, after all components are initialized
logger.info(f"⚡ Rate limits: 10/min, 1200/day")
logger.info("✅ Startup completed successfully")

if __name__ == "__main__":
    import uvicorn
    
    try:
        print("🚀 Starting Enhanced Gemini-Powered Paraphraser...")
        print(f"📡 Using Google Gemini API: {processor.model}")
        print(f"👥 User Rate Limit: {user_limiter.daily_limit_per_user} requests per day")
        print(f"🔄 Queue System: Enabled")
        print(f"⚡ Enhanced Rate Limits: 10/min, 1200/day")
        print("🌐 FastAPI server ready!")
        print("📚 API Documentation: http://127.0.0.1:8000/docs")
        print("📊 Metrics: http://127.0.0.1:8000/metrics")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # Changed to accept external connections
            port=int(os.getenv("PORT", 8000)),  # Use PORT env var for cloud deployment
            reload=False  # Disabled for production
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Server startup failed: {e}")
