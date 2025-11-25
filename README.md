# Skillmotion.AI 🚀

**Your AI-Powered Skill Development Assistant**

A fully functional, scalable, and **100% free-to-deploy** AI assistant web application built with cutting-edge open-source technologies. Skillmotion.AI helps users identify skill gaps, create personalized learning plans, and accelerate career growth through AI-driven guidance and voice-enabled interactions.

## ✨ Features

- **🧠 AI-Powered Analysis**: Advanced LLM technology for skill assessment and gap analysis
- **🎤 Voice Interaction**: Natural voice conversations with Web Speech API and Edge-TTS
- **📊 Personalized Learning Plans**: Custom roadmaps based on user profiles and goals
- **🎯 Career Guidance**: Industry-specific advice and advancement strategies
- **⚡ Real-time Chat**: Smooth, responsive chat interface with typing indicators
- **📱 Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **🔒 Privacy-First**: All processing happens on your infrastructure

## 🛠️ Technology Stack

### Frontend
- **Pure HTML/CSS/JavaScript** - No frameworks, maximum performance
- **Tailwind CSS** - For beautiful, responsive design
- **Web Speech API** - For voice input capabilities
- **Glassmorphism UI** - Modern, elegant design aesthetics

### Backend
- **FastAPI** - High-performance Python web framework
- **LangChain/LangGraph** - Advanced LLM orchestration and workflows
- **Groq API** - Ultra-fast LLM inference (Mixtral/Llama models)
- **Cohere Embeddings** - Semantic search and RAG capabilities
- **Edge-TTS** - High-quality, free text-to-speech synthesis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Free API keys from Groq and Cohere

### 1. Get Free API Keys

**Groq API (Free):**
1. Visit [console.groq.com](https://console.groq.com/)
2. Sign up for a free account
3. Generate your API key

**Cohere API (Free):**
1. Visit [dashboard.cohere.ai](https://dashboard.cohere.ai/)
2. Sign up for a free account  
3. Generate your API key

### 2. Setup and Run

```bash
# Clone or download the project
cd skillmotion_ai

# Update .env file with your API keys
# Edit .env and replace 'your_groq_api_key_here' and 'your_cohere_api_key_here'

# Run the application (this will install dependencies automatically)
python run.py
```

That's it! The application will:
- Install all required dependencies
- Start the backend server on http://localhost:8000
- Start the frontend server on http://localhost:3000
- Open your browser automatically

## 📁 Project Structure

```
skillmotion_ai/
├── frontend/
│   └── index.html           # Landing page + chatbot UI
├── backend/
│   ├── main.py              # FastAPI server
│   ├── chatbot_agent.py     # LangGraph AI agent
│   ├── utils.py             # Async utilities
│   ├── tts.py               # Edge-TTS integration
│   └── __init__.py          # Package init
├── static/
│   └── audio/               # Generated audio files
├── .env                     # Environment variables (add your API keys here)
├── requirements.txt         # Python dependencies
├── run.py                   # Main application runner
└── README.md               # This file
```

## 🤖 AI Agent Capabilities

### Intelligent Workflows
- **Intent Analysis**: Automatically categorizes user requests
- **Skill Assessment**: Evaluates current skill levels and identifies gaps
- **Learning Plan Generation**: Creates personalized development roadmaps
- **Career Guidance**: Provides industry-specific advancement advice
- **Context Awareness**: Maintains conversation memory and user profiles

### Supported Skill Categories
- **Programming**: Python, JavaScript, Web Development, Data Science
- **Soft Skills**: Communication, Leadership, Project Management
- **Technical Skills**: Machine Learning, Cloud Computing, DevOps
- **Career Development**: Interview Prep, Resume Building, Networking

## 🎤 Voice Features

### Input Capabilities
- **Web Speech API**: Real-time voice-to-text conversion
- **Multiple Languages**: Support for various accents and languages
- **Noise Handling**: Robust speech recognition in various environments

### Output Capabilities
- **Edge-TTS Integration**: High-quality voice synthesis
- **Multiple Voices**: Choose from various natural-sounding voices
- **Adjustable Speed**: Customize playback rate for optimal learning
- **Browser Fallback**: Uses browser TTS if Edge-TTS is unavailable

## 🚀 Deployment Options

### Free Deployment Platforms

**1. Render (Recommended)**
```yaml
# render.yaml
services:
  - type: web
    name: skillmotion-ai
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT"
```

**2. Railway**
```bash
railway login
railway new skillmotion-ai
railway add
railway deploy
```

**3. Replit**
1. Import GitHub repository
2. Set environment variables
3. Run `python run.py`

## 🔧 Configuration

### Environment Variables
Edit `.env` file to configure:
- API keys (Groq, Cohere)
- Server settings
- TTS preferences
- Rate limiting
- Cache settings

### Voice Settings
Available voices include:
- English (US): Aria, Jenny, Guy, Davis
- English (UK): Sonia, Ryan
- English (Australia): Natasha
- English (Canada): Clara

## 🧪 Testing

Test the application:
1. Run `python run.py`
2. Open http://localhost:3000
3. Try the chat interface
4. Test voice input/output features
5. Check backend API at http://localhost:8000/docs

## 🔐 Security Features

- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Input Validation**: Sanitizes all user inputs
- **Error Handling**: Graceful handling of edge cases
- **CORS Configuration**: Proper cross-origin resource sharing
- **No Data Persistence**: Conversations are not permanently stored

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Groq** for providing free, ultra-fast LLM inference
- **Cohere** for free embedding services
- **Microsoft** for Edge-TTS technology
- **LangChain** team for the excellent orchestration framework
- **FastAPI** for the outstanding web framework

---

**Built with ❤️ using 100% free and open-source technologies**

*Skillmotion.AI - Empowering careers through AI-driven skill development*