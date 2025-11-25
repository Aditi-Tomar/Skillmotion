import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import shutil

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules - using absolute imports
try:
    from chatbot_agent import SkillmotionAgent
    from utils import async_retry, log_performance
    from tts import EdgeTTSService
    from resume_processor import ResumeProcessor
    from mysql_client import MySQLService
    from session_manager import SessionManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the correct directory")
    raise

app = FastAPI(
    title="Skillmotion.AI API",
    description="AI-powered skill development assistant with voice capabilities and resume analysis",
    version="1.0.0"
)

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services with error handling
skillmotion_agent = None
tts_service = None
resume_processor = None
mysql_service = None
session_manager = None


def initialize_services():
    """Initialize AI services with proper error handling"""
    global skillmotion_agent, tts_service, resume_processor, mysql_service, session_manager

    try:
        # Check if API keys are available
        groq_key = os.getenv("GROQ_API_KEY")
        cohere_key = os.getenv("COHERE_API_KEY")

        if not groq_key or groq_key == "your_groq_api_key_here":
            logger.warning("Groq API key not configured - AI chat will be limited")
            skillmotion_agent = None
        else:
            skillmotion_agent = SkillmotionAgent()
            logger.info("✅ AI Agent initialized successfully")

        if not cohere_key or cohere_key == "your_cohere_api_key_here":
            logger.warning("Cohere API key not configured - embeddings will be limited")

        # TTS service doesn't require API keys (uses Edge-TTS)
        tts_service = EdgeTTSService()
        logger.info("✅ TTS Service initialized successfully")

        # Resume processor
        resume_processor = ResumeProcessor()
        logger.info("✅ Resume Processor initialized successfully")

        try:
            mysql_service = MySQLService()
            # Note: Connection pool is created on first use
            session_manager = SessionManager(mysql_service)
            logger.info("✅ MySQL and Session Manager initialized successfully")
        except Exception as e:
            logger.warning(f"MySQL initialization failed: {e} - Sessions will be in-memory only")
            mysql_service = None
            session_manager = None

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        skillmotion_agent = None
        tts_service = EdgeTTSService()  # TTS should still work
        resume_processor = ResumeProcessor()  # Resume processor should still work


# Initialize services on startup
initialize_services()


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    voice_enabled: bool = False
    resume_data: Optional[Dict[str, Any]] = None
    is_interruption: bool = False
    current_response: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    audio_url: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ResumeUploadResponse(BaseModel):
    success: bool
    message: str
    resume_data: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Skillmotion.AI",
        "ai_enabled": skillmotion_agent is not None,
        "tts_enabled": tts_service is not None,
        "resume_processor_enabled": resume_processor is not None,
        "groq_configured": os.getenv("GROQ_API_KEY") not in [None, "your_groq_api_key_here"],
        "cohere_configured": os.getenv("COHERE_API_KEY") not in [None, "your_cohere_api_key_here"]
    }


# Serve frontend
@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    frontend_path = Path("../frontend/index.html")
    if frontend_path.exists():
        return FileResponse(str(frontend_path.resolve()))
    else:
        # Fallback to root directory
        root_path = Path("../index.html")
        if root_path.exists():
            return FileResponse(str(root_path.resolve()))
        else:
            raise HTTPException(status_code=404, detail="Frontend not found")


# Resume upload endpoint
@app.post("/api/upload-resume", response_model=ResumeUploadResponse)
@log_performance
async def upload_resume(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing resume upload: {file.filename}")

        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain"
        ]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload PDF, DOC, DOCX, or TXT files only."
            )

        # Check file size (limit to 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Please upload files smaller than 10MB."
            )

        if not resume_processor:
            raise HTTPException(
                status_code=503,
                detail="Resume processing service not available"
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

        try:
            # Process the resume
            resume_data = await resume_processor.process_resume(tmp_file_path, file.filename)

            # Generate initial analysis
            analysis = await resume_processor.analyze_resume(resume_data)

            return ResumeUploadResponse(
                success=True,
                message="Resume uploaded and processed successfully",
                resume_data=resume_data,
                analysis=analysis
            )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing resume upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process resume. Please try again."
        )


# Session management endpoints
@app.post("/api/session/start")
async def start_session(user_id: str, mode: str):
    try:
        if not session_manager:
            return {"success": False, "error": "Session manager not available"}

        session = await session_manager.create_session(user_id, mode)
        return {
            "success": True,
            "session_id": session.session_id,
            "mode": session.mode
        }
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/session/end")
async def end_session(session_id: str):
    try:
        if not session_manager:
            return {"success": False, "error": "Session manager not available"}

        await session_manager.end_session(session_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return {"success": False, "error": str(e)}

# Check for exit keywords
def check_exit_keyword(message: str) -> bool:
    exit_keywords = ["bye", "goodbye", "exit", "stop listening", "quit", "close"]
    message_lower = message.lower().strip()
    return any(keyword in message_lower for keyword in exit_keywords)

# Main chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
@log_performance
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Processing chat request from user: {request.user_id}")

        # Check for exit keywords
        if check_exit_keyword(request.message):
            if session_manager:
                session = await session_manager.get_active_session(request.user_id)
                if session:
                    await session_manager.end_session(session.session_id)

            return ChatResponse(
                response="Okay, I'll stop listening now. It was great talking with you! Say 'hey SkillMotion' anytime you need help with your career development.",
                audio_url=None,
                metadata={"should_close": True}
            )

        # Check if AI agent is available
        if skillmotion_agent is None:
            return ChatResponse(
                response="I'm currently running in demo mode. To enable full AI capabilities, please add your free API keys (Groq and Cohere) to the .env file and restart the server. You can get free keys from console.groq.com and dashboard.cohere.ai",
                audio_url=None,
                metadata={"ai_enabled": False}
            )

        # Handle interruption if specified
        if request.is_interruption and session_manager:
            await session_manager.handle_interruption(
                request.user_id,
                request.current_response or "",
                request.message
            )

        # Save user message to session
        if session_manager:
            await session_manager.save_message(
                request.user_id,
                'user',
                request.message,
                {'voice_enabled': request.voice_enabled}
            )

        # Get session context for resume data
        resume_data = request.resume_data
        if session_manager and not resume_data:
            context = await session_manager.get_session_context(request.user_id)
            if context and context.get('resume_data'):
                resume_data = context['resume_data']

        # Process message through AI agent with resume context
        ai_response = await skillmotion_agent.process_message(
            message=request.message,
            user_id=request.user_id,
            resume_data=resume_data
        )

        # Save AI response to session
        if session_manager:
            if session_manager:
                session = await session_manager.get_active_session(request.user_id)
                if session:
                    session.current_speaking = True

            await session_manager.save_message(
                request.user_id,
                'ai',
                ai_response["response"],
                ai_response.get("metadata", {})
            )

        # Generate audio if voice is enabled and TTS is available
        audio_url = None
        if request.voice_enabled and tts_service:
            try:
                audio_url = await tts_service.generate_speech(
                    text=ai_response["response"],
                    user_id=request.user_id
                )
            except Exception as e:
                logger.warning(f"TTS generation failed: {e}")
                audio_url = None

        # Check if there are pending queries
        has_pending = False
        if session_manager:
            pending = await session_manager.process_pending_queries(request.user_id)
            has_pending = len(pending) > 0

        metadata = ai_response.get("metadata", {})
        metadata['has_pending'] = has_pending

        return ChatResponse(
            response=ai_response["response"],
            audio_url=audio_url,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return ChatResponse(
            response="I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question.",
            audio_url=None,
            metadata={"error": str(e)}
        )


# Skill assessment endpoint
@app.post("/api/assess-skills")
@async_retry(max_attempts=3)
async def assess_skills(request: Dict[str, Any]):
    try:
        if skillmotion_agent is None:
            raise HTTPException(status_code=503, detail="AI services not available - check API keys")

        assessment = await skillmotion_agent.assess_skills(request)
        return {"assessment": assessment}
    except Exception as e:
        logger.error(f"Error in skill assessment: {str(e)}")
        raise HTTPException(status_code=500, detail="Assessment failed")


# Learning plan endpoint
@app.post("/api/create-learning-plan")
async def create_learning_plan(request: Dict[str, Any]):
    try:
        if skillmotion_agent is None:
            raise HTTPException(status_code=503, detail="AI services not available - check API keys")

        plan = await skillmotion_agent.create_learning_plan(request)
        return {"learning_plan": plan}
    except Exception as e:
        logger.error(f"Error creating learning plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Plan creation failed")


# Voice synthesis endpoint
@app.post("/api/synthesize-speech")
async def synthesize_speech(request: Dict[str, str]):
    try:
        if tts_service is None:
            raise HTTPException(status_code=503, detail="TTS service not available")

        audio_url = await tts_service.generate_speech(
            text=request["text"],
            user_id=request.get("user_id", "anonymous")
        )
        return {"audio_url": audio_url}
    except Exception as e:
        logger.error(f"Error in speech synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail="Speech synthesis failed")


# Resume analysis endpoint
@app.post("/api/analyze-resume")
async def analyze_resume_endpoint(request: Dict[str, Any]):
    try:
        if not resume_processor:
            raise HTTPException(status_code=503, detail="Resume processor not available")

        resume_data = request.get("resume_data")
        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data required")

        analysis = await resume_processor.analyze_resume(resume_data)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        raise HTTPException(status_code=500, detail="Resume analysis failed")


# Skill gap analysis endpoint
@app.post("/api/skill-gap-analysis")
async def skill_gap_analysis(request: Dict[str, Any]):
    try:
        if not resume_processor or not skillmotion_agent:
            raise HTTPException(status_code=503, detail="Services not available")

        resume_data = request.get("resume_data")
        target_role = request.get("target_role", "")

        if not resume_data:
            raise HTTPException(status_code=400, detail="Resume data required")

        gap_analysis = await resume_processor.identify_skill_gaps(resume_data, target_role)
        return {"skill_gaps": gap_analysis}
    except Exception as e:
        logger.error(f"Error in skill gap analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Skill gap analysis failed")


# Support escalation endpoint
@app.post("/api/notify/support")
async def create_support_request(request: Dict[str, Any]):
    try:
        user_id = request.get("user_id", "anonymous")
        subject = request.get("subject", "Support Request")
        description = request.get("description", "")
        context = request.get("context", {})

        if not description:
            raise HTTPException(status_code=400, detail="Description is required")

        if session_manager and mysql_service:
            support_request = await mysql_service.create_support_request(
                user_id=user_id,
                subject=subject,
                description=description,
                context=context
            )

            if support_request:
                logger.info(f"Support request created for user {user_id}")
                return {
                    "success": True,
                    "message": "Your support request has been received. Our team will contact you at info@skillmotion.ai soon.",
                    "request_id": support_request['id']
                }

        return {
            "success": True,
            "message": "Support request received. Please email us directly at info@skillmotion.ai for assistance."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating support request: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create support request")


# Pending queries endpoint
@app.get("/api/pending-queries/{user_id}")
async def get_pending_queries(user_id: str):
    try:
        if not session_manager:
            return {"pending_queries": []}

        pending = await session_manager.process_pending_queries(user_id)
        return {
            "pending_queries": [
                {
                    "query": q.query,
                    "priority": q.priority,
                    "timestamp": q.timestamp.isoformat(),
                    "query_id": q.query_id
                } for q in pending
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching pending queries: {str(e)}")
        return {"pending_queries": []}


# Serve static files (for audio files)
try:
    # Try multiple possible static directory locations
    static_paths = [
        Path("../static").resolve(),
        Path("static").resolve(),
        Path("../../static").resolve()
    ]

    static_mounted = False
    for static_path in static_paths:
        if static_path.exists():
            app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
            logger.info(f"✅ Mounted static files from: {static_path}")
            static_mounted = True
            break

    if not static_mounted:
        # Create static directory if it doesn't exist
        static_path = Path("../static").resolve()
        static_path.mkdir(parents=True, exist_ok=True)
        (static_path / "audio").mkdir(exist_ok=True)
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        logger.info(f"✅ Created and mounted static files at: {static_path}")

except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=True)
