import os
import time
import asyncio
import aiohttp
import hashlib
from typing import List, Dict, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import json
from dotenv import load_dotenv
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

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
    
    async def add_request(self, request_data, user_id):
        """Add request to queue"""
        queue_item = {
            'request_data': request_data,
            'user_id': user_id,
            'timestamp': time.time(),
            'future': asyncio.Future()
        }
        self.queue.append(queue_item)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self.process_queue())
        
        return queue_item['future']
    
    async def process_queue(self):
        """Process queued requests with rate limiting"""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            while self.queue:
                item = self.queue.popleft()
                
                # Wait for rate limit interval
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                
                if time_since_last < self.min_interval:
                    wait_time = self.min_interval - time_since_last
                    await asyncio.sleep(wait_time)
                
                try:
                    # Process the actual request
                    result = await processor.paraphrase_text(
                        item['request_data'].text,
                        item['request_data'].num_variations
                    )
                    
                    # Add queue info
                    result['queue_position'] = 0
                    result['estimated_wait_time'] = 0.0
                    
                    item['future'].set_result(result)
                    
                except Exception as e:
                    item['future'].set_exception(e)
                
                self.last_request_time = time.time()
                
        finally:
            self.processing = False
    
    def get_queue_status(self):
        """Get current queue status"""
        return {
            'queue_length': len(self.queue),
            'estimated_wait_time': len(self.queue) * self.min_interval,
            'processing': self.processing
        }

class UserRateLimiter:
    def __init__(self):
        # Track user requests by day
        self.user_requests = defaultdict(lambda: {"count": 0, "date": None})
        self.daily_limit_per_user = 10  # Reduced from 14 for safety buffer
        
        # Track failed requests to avoid wasting quota
        self.failed_requests = defaultdict(int)
        self.cleanup_interval = 3600  # Clean up old data every hour
        self.last_cleanup = time.time()

    def cleanup_old_data(self):
        """Clean up old tracking data"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            # Remove old failed request counts
            self.failed_requests.clear()
            self.last_cleanup = current_time

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
        
        today = datetime.now().date()
        user_data = self.user_requests[user_id]
        
        # Reset counter if new day
        if user_data["date"] != today:
            user_data["count"] = 0
            user_data["date"] = today
        
        # Check limit
        if user_data["count"] >= self.daily_limit_per_user:
            return False
        
        # Increment counter
        user_data["count"] += 1
        return True
    
    def get_user_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user today"""
        today = datetime.now().date()
        user_data = self.user_requests[user_id]
        
        if user_data["date"] != today:
            return self.daily_limit_per_user
        
        return max(0, self.daily_limit_per_user - user_data["count"])
    
    def log_failed_request(self, user_id: str):
        """Log failed request (doesn't count against user limit)"""
        self.failed_requests[user_id] += 1

class GeminiProcessor:
    """Process text using Google Gemini API with improved error handling"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
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
        
    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30
            )
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=connector
            )
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
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": 60 - (current_time - self.last_reset),
                    "type": "minute_limit"
                }
            )
        
        if self.daily_requests >= 1200:  # 1200/day instead of 1400 for buffer
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
            
            async with session.post(url, json=payload, headers=headers) as response:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                self.requests_count += 1
                self.daily_requests += 1
                
                if response.status == 200:
                    result = await response.json()
                    
                    if 'candidates' in result and result['candidates']:
                        content = result['candidates'][0]['content']['parts'][0]['text']
                        
                        tokens_used = None
                        if 'usageMetadata' in result:
                            tokens_used = result['usageMetadata'].get('totalTokenCount', 0)
                        
                        self.success_count += 1
                        
                        return {
                            'content': content.strip(),
                            'tokens_used': tokens_used,
                            'success': True,
                            'response_time': response_time
                        }
                    else:
                        self.error_count += 1
                        return {'success': False, 'error': 'No content generated'}
                
                elif response.status == 429:
                    self.error_count += 1
                    error_text = await response.text()
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
                    return {'success': False, 'error': f"API error {response.status}: {error_text}"}
                    
        except HTTPException:
            raise
        except Exception as e:
            self.error_count += 1
            logger.error(f"Gemini API error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def paraphrase_text(self, text: str, num_variations: int = 3) -> Dict:
        """Generate paraphrases using Gemini with retry logic"""
        start_time = time.time()
        
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
                
                return {
                    'original_text': text,
                    'paraphrases': paraphrases[:num_variations],
                    'total_variations': len(paraphrases),
                    'processing_time': round(processing_time, 3),
                    'service_used': f'Google Gemini {self.model}',
                    'tokens_used': result.get('tokens_used')
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Paraphrasing failed: {result.get('error', 'Unknown error')}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Paraphrasing error: {e}")
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

# Initialize components
processor = GeminiProcessor()
user_limiter = UserRateLimiter()
request_queue = RequestQueue()

# Dependency functions
async def check_user_rate_limit(request: Request):
    user_id = user_limiter.get_user_id(request)
    
    if not user_limiter.check_user_limit(user_id):
        remaining = user_limiter.get_user_remaining_requests(user_id)
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

@app.get("/health")
async def health_check():
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
        "avg_response_time": perf_stats["avg_response_time"]
    }

@app.get("/usage")
async def usage_stats():
    """Get detailed usage statistics"""
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

@app.get("/user-status")
async def user_status(request: Request):
    """Get user's current rate limit status"""
    user_id = user_limiter.get_user_id(request)
    return {
        "user_id": user_id,
        "daily_limit": user_limiter.daily_limit_per_user,
        "requests_remaining": user_limiter.get_user_remaining_requests(user_id),
        "failed_requests": user_limiter.failed_requests.get(user_id, 0),
        "reset_time": "00:00 UTC tomorrow"
    }

@app.get("/queue-status")
async def queue_status():
    """Get current queue status"""
    return request_queue.get_queue_status()

@app.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase_endpoint(
    request_data: ParaphraseRequest,
    request: Request,
    user_id: str = Depends(check_user_rate_limit)
):
    """Generate paraphrases of the provided text with queue management"""
    try:
        # Add to queue and wait for processing
        result = await request_queue.add_request(request_data, user_id)
        
        # Add user info to response
        result["user_requests_remaining"] = user_limiter.get_user_remaining_requests(user_id)
        
        return ParaphraseResponse(**result)
        
    except Exception as e:
        # Log failed request but don't count against user limit
        user_limiter.log_failed_request(user_id)
        logger.error(f"Request failed for user {user_id}: {e}")
        raise

@app.get("/metrics")
async def metrics():
    """Get comprehensive metrics for monitoring"""
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
            "total_users_served": len(user_limiter.user_requests)
        }
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    await processor.close()
    logger.info("Application shutdown complete")

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("🚀 Enhanced Gemini Paraphraser API starting...")
    logger.info(f"📡 Using model: {processor.model}")
    logger.info(f"👥 User limit: {user_limiter.daily_limit_per_user} requests/day")
    logger.info(f"🔄 Queue system: Enabled")
    logger.info(f"⚡ Rate limits: 10/min, 1200/day")

if __name__ == "__main__":
    import uvicorn
    
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