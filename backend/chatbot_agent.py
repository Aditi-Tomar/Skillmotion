import asyncio
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
import os
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Cohere for embeddings
import cohere

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    user_id: str
    skills: List[str] = None
    career_goals: List[str] = None
    experience_level: str = "beginner"
    learning_preferences: Dict[str, Any] = None
    assessment_history: List[Dict] = None
    resume_data: Optional[Dict[str, Any]] = None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: UserProfile
    context: Dict[str, Any]
    current_step: str


class SkillmotionAgent:
    def __init__(self):
        # Get API keys
        groq_api_key = os.getenv("GROQ_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY in your .env file")

        # Initialize LLM with Groq
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",  # Free tier model
            temperature=0.7,
            max_tokens=1024
        )

        # Initialize Cohere for embeddings (optional)
        self.cohere_client = None
        if cohere_api_key and cohere_api_key != "your_cohere_api_key_here":
            try:
                self.cohere_client = cohere.Client(cohere_api_key)
                logger.info("✅ Cohere client initialized")
            except Exception as e:
                logger.warning(f"Cohere initialization failed: {e}")
        else:
            logger.warning("Cohere API key not configured - embeddings disabled")

        # Memory for conversations
        self.conversation_memory = {}

        # User profiles storage
        self.user_profiles = {}

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()

        # Skill database (in production, this would be a vector database)
        self.skill_database = self._initialize_skill_database()

    def _initialize_skill_database(self) -> Dict[str, Dict]:
        """Initialize a comprehensive skill database with categories and learning paths"""
        return {
            "programming": {
                "python": {
                    "description": "High-level programming language for web development, data science, and automation",
                    "prerequisites": ["basic_computer_literacy"],
                    "learning_path": ["syntax_basics", "data_structures", "functions", "oop", "frameworks"],
                    "assessment_questions": [
                        "What is the difference between lists and tuples in Python?",
                        "How do you handle exceptions in Python?",
                        "Explain Python's GIL and its implications"
                    ]
                },
                "javascript": {
                    "description": "Programming language for web development and modern applications",
                    "prerequisites": ["html", "css"],
                    "learning_path": ["syntax", "dom_manipulation", "async_programming", "frameworks", "node_js"],
                    "assessment_questions": [
                        "What is the event loop in JavaScript?",
                        "Explain closures and their use cases",
                        "What are Promises and how do they work?"
                    ]
                }
            },
            "data_science": {
                "machine_learning": {
                    "description": "Field of AI that enables computers to learn without explicit programming",
                    "prerequisites": ["python", "mathematics", "statistics"],
                    "learning_path": ["supervised_learning", "unsupervised_learning", "deep_learning",
                                      "model_deployment"],
                    "assessment_questions": [
                        "What is overfitting and how can you prevent it?",
                        "Explain the bias-variance tradeoff",
                        "What is cross-validation and why is it important?"
                    ]
                }
            },
            "soft_skills": {
                "communication": {
                    "description": "Ability to convey information effectively and efficiently",
                    "prerequisites": [],
                    "learning_path": ["active_listening", "public_speaking", "written_communication",
                                      "non_verbal_communication"],
                    "assessment_questions": [
                        "How do you handle difficult conversations?",
                        "What strategies do you use for effective team communication?",
                        "How do you adapt your communication style for different audiences?"
                    ]
                }
            }
        }

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for processing user interactions"""

        def analyze_intent(state: AgentState) -> AgentState:
            """Analyze user intent from their message"""
            last_message = state["messages"][-1].content

            # Enhanced intent classification with resume-specific intents
            intents = {
                "resume_analysis": ["analyze resume", "review my resume", "resume feedback", "resume review"],
                "skill_gap_analysis": ["skill gaps", "missing skills", "what skills do i need", "skill requirements"],
                "skill_profile_creation": ["create profile", "skill profile", "build profile", "my skills"],
                "learning_plan": ["learning plan", "study plan", "roadmap", "how to learn"],
                "skill_assessment": ["assess", "evaluate", "test", "check my skills", "quiz", "assessment"],
                "career_advice": ["career", "job", "profession", "advancement", "career path"],
                "recommendations": ["recommend", "suggest", "what should i", "help me find"],
                "support": ["help", "support", "issue", "problem", "contact", "assistance", "escalate"],
                "general_chat": ["hello", "hi", "how are you", "what can you do"]
            }

            detected_intent = "general_chat"
            for intent, keywords in intents.items():
                if any(keyword in last_message.lower() for keyword in keywords):
                    detected_intent = intent
                    break

            state["context"]["intent"] = detected_intent
            state["current_step"] = detected_intent
            return state

        def handle_resume_analysis(state: AgentState) -> AgentState:
            """Handle resume analysis requests"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            if not resume_data:
                response = "I'd be happy to analyze your resume! Please upload your resume first using the upload button, and then I can provide detailed feedback on structure, content, ATS compatibility, and optimization suggestions."
            else:
                # Generate detailed resume analysis
                analysis_prompt = f"""
                You are an expert resume reviewer and career counselor. Analyze this resume data and provide comprehensive feedback.

                Resume Data Summary:
                - Skills found: {resume_data.get('skills', {})}
                - Experience entries: {len(resume_data.get('experience', []))}
                - Education entries: {len(resume_data.get('education', []))}
                - Sections present: {resume_data.get('metadata', {}).get('sections_found', [])}
                - Word count: {resume_data.get('metadata', {}).get('word_count', 0)}

                User request: {user_message}

                Provide detailed analysis covering:
                1. Overall structure and organization
                2. Content quality and completeness
                3. Skills presentation and relevance
                4. ATS (Applicant Tracking System) compatibility
                5. Specific recommendations for improvement
                6. Strengths to highlight

                Be specific, actionable, and encouraging in your feedback.
                """

                response = self.llm.invoke([SystemMessage(content=analysis_prompt)])
                response = response.content

            state["messages"].append(AIMessage(content=response))
            return state

        def handle_skill_gap_analysis(state: AgentState) -> AgentState:
            """Handle skill gap analysis requests"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            if not resume_data:
                response = "To identify your skill gaps, I'll need to analyze your resume first. Please upload your resume, and then I can compare your current skills with industry requirements and target role expectations."
            else:
                gap_analysis_prompt = f"""
                You are a career development expert specializing in skill gap analysis. Based on the resume data, identify skill gaps and provide recommendations.

                Current Skills from Resume:
                {json.dumps(resume_data.get('skills', {}), indent=2)}

                Experience Level: {resume_data.get('experience', [])}
                Education: {resume_data.get('education', [])}

                User request: {user_message}

                Provide analysis including:
                1. Current skill strengths and areas of expertise
                2. Identified skill gaps for career advancement
                3. Industry trends and in-demand skills
                4. Prioritized learning recommendations
                5. Specific resources and learning paths
                6. Timeline suggestions for skill development

                Focus on actionable insights and practical next steps.
                """

                response = self.llm.invoke([SystemMessage(content=gap_analysis_prompt)])
                response = response.content

            state["messages"].append(AIMessage(content=response))
            return state

        def handle_skill_profile_creation(state: AgentState) -> AgentState:
            """Handle skill profile creation requests"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            profile_prompt = f"""
            You are a career development specialist helping create comprehensive skill profiles.

            Available Data:
            - Resume data: {resume_data is not None}
            - Current skills: {user_profile.skills or 'Not specified'}
            - Career goals: {user_profile.career_goals or 'Not specified'}
            - Experience level: {user_profile.experience_level}

            User request: {user_message}

            Create a comprehensive skill profile including:
            1. Technical skills assessment and categorization
            2. Soft skills evaluation
            3. Industry knowledge and domain expertise
            4. Certifications and credentials
            5. Skill proficiency levels (beginner, intermediate, advanced)
            6. Skill development priorities
            7. Career trajectory recommendations

            If resume data is available, use it to provide detailed insights. Otherwise, guide the user through profile creation questions.
            """

            response = self.llm.invoke([SystemMessage(content=profile_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        def handle_learning_plan(state: AgentState) -> AgentState:
            """Create personalized learning plans"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            plan_prompt = f"""
            Create a personalized learning plan based on:
            User request: {user_message}
            Current skills: {user_profile.skills or 'Not specified'}
            Career goals: {user_profile.career_goals or 'Not specified'}
            Experience level: {user_profile.experience_level}
            Resume data available: {resume_data is not None}

            If resume data is available, use the extracted skills and experience to create a more targeted plan.

            Provide a structured learning plan with:
            1. Learning objectives and goals
            2. Skill priorities and focus areas
            3. Recommended learning resources (courses, books, tutorials)
            4. Practical projects and hands-on exercises
            5. Timeline and milestones (weekly/monthly goals)
            6. Assessment and progress tracking methods
            7. Community and networking opportunities

            Focus on free and accessible resources when possible.
            """

            response = self.llm.invoke([SystemMessage(content=plan_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        def handle_skill_assessment(state: AgentState) -> AgentState:
            """Handle skill assessment requests"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            assessment_prompt = f"""
            You are a skill assessment specialist. Create interactive assessments based on the user's background.

            User background:
            - Resume data: {resume_data is not None}
            - Current skills: {user_profile.skills or 'Not specified'}
            - Experience level: {user_profile.experience_level}

            User request: {user_message}

            Provide skill assessment including:
            1. Assessment methodology and approach
            2. Key skill areas to evaluate
            3. Sample assessment questions (multiple choice, practical scenarios)
            4. Self-evaluation frameworks
            5. Skill rating scales and benchmarks
            6. Next steps based on assessment results

            Make assessments interactive and engaging while being thorough and accurate.
            """

            response = self.llm.invoke([SystemMessage(content=assessment_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        def handle_recommendations(state: AgentState) -> AgentState:
            """Provide personalized recommendations"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            recommendations_prompt = f"""
            You are a career advisor providing personalized recommendations.

            User profile:
            - Resume data: {resume_data is not None}
            - Skills: {user_profile.skills or 'Not specified'}
            - Career goals: {user_profile.career_goals or 'Not specified'}
            - Experience level: {user_profile.experience_level}

            User request: {user_message}

            Provide personalized recommendations for:
            1. Job opportunities and career paths
            2. Skill development courses and certifications
            3. Professional networking opportunities
            4. Industry events and conferences
            5. Books, blogs, and learning resources
            6. Tools and technologies to explore
            7. Portfolio and project ideas

            Tailor recommendations to the user's specific background and goals.
            """

            response = self.llm.invoke([SystemMessage(content=recommendations_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        def handle_career_advice(state: AgentState) -> AgentState:
            """Provide career guidance and advice"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]
            resume_data = user_profile.resume_data

            career_prompt = f"""
            Provide career advice based on:
            User question: {user_message}
            Current skills: {user_profile.skills or 'Not specified'}
            Career goals: {user_profile.career_goals or 'Not specified'}
            Experience level: {user_profile.experience_level}
            Resume data available: {resume_data is not None}

            Give practical, actionable advice for career advancement.
            Focus on skill development opportunities and industry trends.
            Include specific steps the user can take immediately.
            """

            response = self.llm.invoke([SystemMessage(content=career_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        def handle_general_chat(state: AgentState) -> AgentState:
            """Handle general conversation and introductions"""
            user_message = state["messages"][-1].content
            user_profile = state["user_profile"]

            chat_prompt = f"""
            You are Skillmotion.AI, a friendly AI assistant specializing in skill development and career growth.

            User message: {user_message}
            Resume uploaded: {user_profile.resume_data is not None}

            Respond helpfully and guide the conversation toward understanding their skill development needs.

            Available features to mention:
            - Resume upload and analysis
            - Skill gap identification
            - Personalized learning plans
            - Skill assessments
            - Career recommendations
            - Voice interaction support

            Ask engaging questions about their career goals, current skills, or learning interests.
            Keep responses concise but informative.
            """

            response = self.llm.invoke([SystemMessage(content=chat_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        def handle_support(state: AgentState) -> AgentState:
            """Handle support and escalation requests"""
            user_message = state["messages"][-1].content

            support_prompt = f"""
            You are a helpful support assistant for Skillmotion.AI.

            User request: {user_message}

            Acknowledge the user's request and let them know:
            1. Their issue has been noted and will be forwarded to the support team
            2. They can also email info@skillmotion.ai directly for urgent matters
            3. A support specialist will review their request and respond soon
            4. In the meantime, ask if there's anything else you can help with

            Be empathetic and reassuring. Keep the response brief and supportive.
            """

            response = self.llm.invoke([SystemMessage(content=support_prompt)])
            state["messages"].append(AIMessage(content=response.content))
            return state

        # Create the workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_intent", analyze_intent)
        workflow.add_node("resume_analysis", handle_resume_analysis)
        workflow.add_node("skill_gap_analysis", handle_skill_gap_analysis)
        workflow.add_node("skill_profile_creation", handle_skill_profile_creation)
        workflow.add_node("learning_plan", handle_learning_plan)
        workflow.add_node("skill_assessment", handle_skill_assessment)
        workflow.add_node("recommendations", handle_recommendations)
        workflow.add_node("career_advice", handle_career_advice)
        workflow.add_node("support", handle_support)
        workflow.add_node("general_chat", handle_general_chat)

        # Define edges
        workflow.set_entry_point("analyze_intent")

        def route_intent(state: AgentState) -> str:
            return state["current_step"]

        workflow.add_conditional_edges(
            "analyze_intent",
            route_intent,
            {
                "resume_analysis": "resume_analysis",
                "skill_gap_analysis": "skill_gap_analysis",
                "skill_profile_creation": "skill_profile_creation",
                "learning_plan": "learning_plan",
                "skill_assessment": "skill_assessment",
                "recommendations": "recommendations",
                "career_advice": "career_advice",
                "support": "support",
                "general_chat": "general_chat"
            }
        )

        # All paths end after processing
        workflow.add_edge("resume_analysis", END)
        workflow.add_edge("skill_gap_analysis", END)
        workflow.add_edge("skill_profile_creation", END)
        workflow.add_edge("learning_plan", END)
        workflow.add_edge("skill_assessment", END)
        workflow.add_edge("recommendations", END)
        workflow.add_edge("career_advice", END)
        workflow.add_edge("support", END)
        workflow.add_edge("general_chat", END)

        return workflow.compile()

    async def process_message(self, message: str, user_id: str, resume_data: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """Process a user message through the AI agent workflow"""
        try:
            # Get or create user profile
            user_profile = self.user_profiles.get(user_id, UserProfile(user_id=user_id))

            # Update user profile with resume data if provided
            if resume_data:
                user_profile.resume_data = resume_data
                # Extract skills from resume data if available
                if 'skills' in resume_data:
                    extracted_skills = []
                    for category, skills in resume_data['skills'].items():
                        extracted_skills.extend(skills)
                    user_profile.skills = extracted_skills

            # Get conversation memory
            if user_id not in self.conversation_memory:
                self.conversation_memory[user_id] = ConversationBufferWindowMemory(
                    k=10,  # Remember last 10 exchanges
                    return_messages=True
                )

            memory = self.conversation_memory[user_id]

            # Create initial state
            initial_state = AgentState(
                messages=[HumanMessage(content=message)],
                user_profile=user_profile,
                context={"intent": None, "timestamp": datetime.now().isoformat()},
                current_step="analyze_intent"
            )

            # Process through workflow
            result = await asyncio.to_thread(self.workflow.invoke, initial_state)

            # Get the AI response
            ai_response = result["messages"][-1].content

            # Update memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(ai_response)

            # Update user profile if needed
            await self._update_user_profile(user_profile, message, result["context"])
            self.user_profiles[user_id] = user_profile

            return {
                "response": ai_response,
                "metadata": {
                    "intent": result["context"].get("intent"),
                    "timestamp": result["context"].get("timestamp"),
                    "user_profile_updated": True,
                    "resume_data_available": user_profile.resume_data is not None
                }
            }

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question.",
                "metadata": {"error": str(e)}
            }

    async def _update_user_profile(self, profile: UserProfile, message: str, context: Dict):
        """Update user profile based on conversation context"""
        # Extract skills mentioned in conversation
        message_lower = message.lower()

        # Simple skill extraction (in production, use NLP/NER)
        mentioned_skills = []
        for category in self.skill_database.values():
            for skill_name in category.keys():
                if skill_name in message_lower:
                    mentioned_skills.append(skill_name)

        if mentioned_skills:
            if profile.skills is None:
                profile.skills = []
            profile.skills.extend([skill for skill in mentioned_skills if skill not in profile.skills])

    async def assess_skills(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed skill assessment"""
        skills_to_assess = request.get("skills", [])
        user_id = request.get("user_id", "anonymous")

        assessment_results = {}

        for skill in skills_to_assess:
            # Find skill in database
            skill_info = self._find_skill_info(skill)
            if skill_info:
                # Generate assessment questions
                questions = skill_info.get("assessment_questions", [])
                assessment_results[skill] = {
                    "questions": questions,
                    "learning_path": skill_info.get("learning_path", []),
                    "prerequisites": skill_info.get("prerequisites", [])
                }

        return assessment_results

    async def create_learning_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive learning plan"""
        target_skills = request.get("target_skills", [])
        current_skills = request.get("current_skills", [])
        timeline = request.get("timeline", "3 months")

        # Use Cohere embeddings to find related skills and resources if available
        if self.cohere_client:
            try:
                # Create embeddings for skill descriptions
                skill_descriptions = []
                skill_names = []

                for category in self.skill_database.values():
                    for skill_name, skill_info in category.items():
                        skill_descriptions.append(skill_info["description"])
                        skill_names.append(skill_name)

                if skill_descriptions:
                    embeddings = self.cohere_client.embed(
                        texts=skill_descriptions,
                        model="embed-english-light-v3.0"  # Free tier model
                    )

                    # Find related skills based on embeddings similarity
                    # (Simplified implementation - in production, use vector database)

            except Exception as e:
                logger.warning(f"Cohere embeddings failed: {e}")

        # Generate learning plan using LLM
        plan_prompt = f"""
        Create a detailed learning plan for:
        Target skills: {target_skills}
        Current skills: {current_skills}
        Timeline: {timeline}

        Structure the plan with:
        1. Week-by-week breakdown
        2. Specific learning objectives
        3. Recommended free resources
        4. Practice exercises
        5. Milestone checkpoints
        """

        response = await asyncio.to_thread(
            self.llm.invoke,
            [SystemMessage(content=plan_prompt)]
        )

        return {
            "plan": response.content,
            "timeline": timeline,
            "target_skills": target_skills
        }

    def _find_skill_info(self, skill_name: str) -> Optional[Dict]:
        """Find skill information in the database"""
        skill_name_lower = skill_name.lower()

        for category in self.skill_database.values():
            for name, info in category.items():
                if name.lower() == skill_name_lower:
                    return info

        return None