import asyncio
import logging
from typing import Dict, List, Any, Optional
import re
import json
from pathlib import Path
import tempfile
import os
from datetime import datetime

# PDF processing
try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Document processing
try:
    import docx

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ResumeProcessor:
    """Service for processing and analyzing resumes"""

    def __init__(self):
        self.skill_keywords = self._load_skill_keywords()
        self.section_patterns = self._load_section_patterns()

    def _load_skill_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive skill keywords database"""
        return {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                "php", "ruby", "swift", "kotlin", "scala", "r", "matlab", "perl",
                "shell", "bash", "powershell", "sql", "html", "css", "sass", "less"
            ],
            "frameworks_libraries": [
                "react", "angular", "vue", "node.js", "express", "django", "flask",
                "spring", "laravel", "rails", "asp.net", "tensorflow", "pytorch",
                "pandas", "numpy", "scikit-learn", "opencv", "bootstrap", "jquery",
                "redux", "graphql", "rest", "api", "microservices"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
                "oracle", "sql server", "sqlite", "dynamodb", "firebase", "neo4j"
            ],
            "cloud_platforms": [
                "aws", "azure", "google cloud", "gcp", "heroku", "digitalocean",
                "kubernetes", "docker", "terraform", "ansible", "jenkins", "gitlab ci",
                "github actions", "circleci", "travis ci"
            ],
            "tools_technologies": [
                "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
                "trello", "asana", "figma", "sketch", "adobe", "photoshop", "illustrator",
                "linux", "windows", "macos", "vim", "vscode", "intellij", "eclipse"
            ],
            "data_science": [
                "machine learning", "deep learning", "artificial intelligence", "ai",
                "data analysis", "data visualization", "statistics", "big data",
                "hadoop", "spark", "tableau", "power bi", "jupyter", "r studio",
                "nlp", "computer vision", "neural networks"
            ],
            "soft_skills": [
                "leadership", "communication", "teamwork", "problem solving",
                "project management", "agile", "scrum", "kanban", "time management",
                "critical thinking", "creativity", "adaptability", "mentoring",
                "public speaking", "presentation", "negotiation", "collaboration"
            ],
            "certifications": [
                "aws certified", "azure certified", "google cloud certified",
                "pmp", "scrum master", "cissp", "comptia", "cisco", "microsoft certified",
                "oracle certified", "salesforce certified", "kubernetes certified"
            ]
        }

    def _load_section_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying resume sections"""
        return {
            "experience": [
                r"work experience", r"professional experience", r"employment history",
                r"career history", r"work history", r"experience", r"employment"
            ],
            "education": [
                r"education", r"academic background", r"qualifications",
                r"academic qualifications", r"educational background"
            ],
            "skills": [
                r"skills", r"technical skills", r"core competencies",
                r"key skills", r"expertise", r"proficiencies", r"technologies"
            ],
            "projects": [
                r"projects", r"key projects", r"notable projects",
                r"personal projects", r"academic projects", r"portfolio"
            ],
            "certifications": [
                r"certifications", r"certificates", r"professional certifications",
                r"licenses", r"credentials", r"achievements"
            ],
            "summary": [
                r"summary", r"profile", r"objective", r"career objective",
                r"professional summary", r"about", r"overview"
            ]
        }

    async def process_resume(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process uploaded resume file and extract structured data"""
        try:
            # Extract text based on file type
            text_content = await self._extract_text(file_path, filename)

            if not text_content:
                raise ValueError("Could not extract text from resume")

            # Parse resume structure
            parsed_data = await self._parse_resume_structure(text_content)

            # Extract skills
            skills = await self._extract_skills(text_content)

            # Extract experience
            experience = await self._extract_experience(text_content)

            # Extract education
            education = await self._extract_education(text_content)

            # Extract contact information
            contact_info = await self._extract_contact_info(text_content)

            resume_data = {
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "raw_text": text_content,
                "contact_info": contact_info,
                "skills": skills,
                "experience": experience,
                "education": education,
                "sections": parsed_data,
                "metadata": {
                    "word_count": len(text_content.split()),
                    "character_count": len(text_content),
                    "sections_found": list(parsed_data.keys())
                }
            }

            logger.info(f"Successfully processed resume: {filename}")
            return resume_data

        except Exception as e:
            logger.error(f"Error processing resume {filename}: {str(e)}")
            raise

    async def _extract_text(self, file_path: str, filename: str) -> str:
        """Extract text from various file formats"""
        file_extension = Path(filename).suffix.lower()

        try:
            if file_extension == '.pdf':
                return await self._extract_pdf_text(file_path)
            elif file_extension in ['.doc', '.docx']:
                return await self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                return await self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ValueError("PDF processing not available. Install PyPDF2: pip install PyPDF2")

        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise ValueError("Could not read PDF file")

        return text.strip()

    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ValueError("DOCX processing not available. Install python-docx: pip install python-docx")

        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error reading DOCX: {str(e)}")
            raise ValueError("Could not read DOCX file")

        return text.strip()

    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except Exception as e:
            logger.error(f"Error reading TXT: {str(e)}")
            raise ValueError("Could not read TXT file")

        return text.strip()

    async def _parse_resume_structure(self, text: str) -> Dict[str, str]:
        """Parse resume into structured sections"""
        sections = {}
        text_lower = text.lower()

        for section_name, patterns in self.section_patterns.items():
            section_content = ""

            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    # Find the section content
                    start_pos = matches[0].end()

                    # Find the end of this section (start of next section or end of text)
                    end_pos = len(text)
                    for other_section, other_patterns in self.section_patterns.items():
                        if other_section != section_name:
                            for other_pattern in other_patterns:
                                other_matches = list(re.finditer(other_pattern, text_lower[start_pos:]))
                                if other_matches:
                                    potential_end = start_pos + other_matches[0].start()
                                    if potential_end < end_pos:
                                        end_pos = potential_end

                    section_content = text[start_pos:end_pos].strip()
                    break

            if section_content:
                sections[section_name] = section_content

        return sections

    async def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from resume text"""
        found_skills = {}
        text_lower = text.lower()

        for category, keywords in self.skill_keywords.items():
            category_skills = []

            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    category_skills.append(keyword)

            if category_skills:
                found_skills[category] = category_skills

        return found_skills

    async def _extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience from resume"""
        experience_entries = []

        # Look for common experience patterns
        patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|\w+)\s*[:\-]?\s*([^\n]+)',  # Year range with title
            r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4}|\w+)\s*[:\-]?\s*([^\n]+)',  # Month Year format
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                start_date = match.group(1)
                end_date = match.group(2)
                title_company = match.group(3).strip()

                experience_entries.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "title_company": title_company,
                    "raw_text": match.group(0)
                })

        return experience_entries

    async def _extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education information from resume"""
        education_entries = []

        # Common degree patterns
        degree_patterns = [
            r'(bachelor|master|phd|doctorate|associate|diploma|certificate)[\s\w]*in\s+([^\n,]+)',
            r'(b\.?s\.?|m\.?s\.?|m\.?a\.?|ph\.?d\.?|b\.?a\.?)\s+([^\n,]+)',
            r'(university|college|institute|school)\s+of\s+([^\n,]+)'
        ]

        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                education_entries.append({
                    "degree_type": match.group(1),
                    "field_of_study": match.group(2).strip(),
                    "raw_text": match.group(0)
                })

        return education_entries

    async def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from resume"""
        contact_info = {
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
            "website": None
        }

        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info["email"] = email_match.group(0)

        # Phone pattern
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info["phone"] = phone_match.group(0)

        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/([A-Za-z0-9-]+)'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info["linkedin"] = linkedin_match.group(0)

        # GitHub pattern
        github_pattern = r'github\.com/([A-Za-z0-9-]+)'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info["github"] = github_match.group(0)

        # Website pattern
        website_pattern = r'https?://[^\s]+'
        website_match = re.search(website_pattern, text)
        if website_match:
            contact_info["website"] = website_match.group(0)

        return contact_info

    async def analyze_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive resume analysis"""
        try:
            analysis = {
                "overall_score": 0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "skill_analysis": {},
                "structure_analysis": {},
                "content_analysis": {},
                "ats_compatibility": {}
            }

            # Analyze skills
            skills_analysis = await self._analyze_skills(resume_data.get("skills", {}))
            analysis["skill_analysis"] = skills_analysis

            # Analyze structure
            structure_analysis = await self._analyze_structure(resume_data)
            analysis["structure_analysis"] = structure_analysis

            # Analyze content quality
            content_analysis = await self._analyze_content(resume_data)
            analysis["content_analysis"] = content_analysis

            # ATS compatibility check
            ats_analysis = await self._analyze_ats_compatibility(resume_data)
            analysis["ats_compatibility"] = ats_analysis

            # Calculate overall score
            analysis["overall_score"] = await self._calculate_overall_score(analysis)

            # Generate recommendations
            analysis["recommendations"] = await self._generate_recommendations(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            raise

    async def _analyze_skills(self, skills: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze the skills section of the resume"""
        total_skills = sum(len(skill_list) for skill_list in skills.values())

        skill_categories = len(skills.keys())

        # Assess skill diversity
        diversity_score = min(skill_categories * 20, 100)  # Max 100 for 5+ categories

        # Assess technical depth
        technical_categories = ["programming_languages", "frameworks_libraries", "databases", "cloud_platforms"]
        technical_skills = sum(len(skills.get(cat, [])) for cat in technical_categories)
        technical_score = min(technical_skills * 5, 100)  # Max 100 for 20+ technical skills

        return {
            "total_skills": total_skills,
            "skill_categories": skill_categories,
            "diversity_score": diversity_score,
            "technical_score": technical_score,
            "skills_by_category": skills,
            "recommendations": []
        }

    async def _analyze_structure(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resume structure and organization"""
        sections = resume_data.get("sections", {})
        required_sections = ["experience", "education", "skills"]
        optional_sections = ["summary", "projects", "certifications"]

        has_required = sum(1 for section in required_sections if section in sections)
        has_optional = sum(1 for section in optional_sections if section in sections)

        structure_score = (has_required / len(required_sections)) * 70 + (has_optional / len(optional_sections)) * 30

        return {
            "structure_score": structure_score,
            "required_sections_present": has_required,
            "optional_sections_present": has_optional,
            "total_sections": len(sections),
            "missing_sections": [s for s in required_sections if s not in sections],
            "present_sections": list(sections.keys())
        }

    async def _analyze_content(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content quality and completeness"""
        raw_text = resume_data.get("raw_text", "")
        word_count = len(raw_text.split())

        # Optimal word count is typically 400-800 words
        if 400 <= word_count <= 800:
            length_score = 100
        elif word_count < 400:
            length_score = (word_count / 400) * 100
        else:
            length_score = max(100 - ((word_count - 800) / 10), 50)

        # Check for action verbs
        action_verbs = [
            "achieved", "managed", "led", "developed", "created", "implemented",
            "improved", "increased", "reduced", "optimized", "designed", "built"
        ]

        action_verb_count = sum(1 for verb in action_verbs if verb in raw_text.lower())
        action_verb_score = min(action_verb_count * 10, 100)

        # Check for quantifiable achievements
        number_pattern = r'\d+%|\$\d+|\d+\+|increased by \d+|reduced by \d+'
        quantifiable_achievements = len(re.findall(number_pattern, raw_text, re.IGNORECASE))
        quantifiable_score = min(quantifiable_achievements * 20, 100)

        return {
            "word_count": word_count,
            "length_score": length_score,
            "action_verb_count": action_verb_count,
            "action_verb_score": action_verb_score,
            "quantifiable_achievements": quantifiable_achievements,
            "quantifiable_score": quantifiable_score
        }

    async def _analyze_ats_compatibility(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ATS (Applicant Tracking System) compatibility"""
        raw_text = resume_data.get("raw_text", "")

        # Check for standard section headers
        standard_headers = ["experience", "education", "skills", "summary"]
        header_score = sum(1 for header in standard_headers if header in raw_text.lower()) * 25

        # Check for consistent formatting
        bullet_points = raw_text.count("•") + raw_text.count("-") + raw_text.count("*")
        formatting_score = min(bullet_points * 5, 100)

        # Check for keywords density
        total_words = len(raw_text.split())
        skill_words = sum(len(skills) for skills in resume_data.get("skills", {}).values())
        keyword_density = (skill_words / total_words) * 100 if total_words > 0 else 0
        keyword_score = min(keyword_density * 10, 100)

        ats_score = (header_score + formatting_score + keyword_score) / 3

        return {
            "ats_score": ats_score,
            "header_score": header_score,
            "formatting_score": formatting_score,
            "keyword_score": keyword_score,
            "keyword_density": keyword_density
        }

    async def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall resume score"""
        weights = {
            "skill_analysis": 0.3,
            "structure_analysis": 0.25,
            "content_analysis": 0.25,
            "ats_compatibility": 0.2
        }

        total_score = 0
        for category, weight in weights.items():
            if category in analysis:
                if category == "skill_analysis":
                    score = (analysis[category]["diversity_score"] + analysis[category]["technical_score"]) / 2
                elif category == "structure_analysis":
                    score = analysis[category]["structure_score"]
                elif category == "content_analysis":
                    content = analysis[category]
                    score = (content["length_score"] + content["action_verb_score"] + content["quantifiable_score"]) / 3
                elif category == "ats_compatibility":
                    score = analysis[category]["ats_score"]

                total_score += score * weight

        return round(total_score, 1)

    async def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations based on analysis"""
        recommendations = []

        # Structure recommendations
        structure = analysis.get("structure_analysis", {})
        if structure.get("structure_score", 0) < 70:
            missing = structure.get("missing_sections", [])
            if missing:
                recommendations.append(f"Add missing sections: {', '.join(missing)}")

        # Content recommendations
        content = analysis.get("content_analysis", {})
        if content.get("word_count", 0) < 400:
            recommendations.append("Expand your resume content. Aim for 400-800 words total.")
        elif content.get("word_count", 0) > 800:
            recommendations.append("Consider condensing your resume. Keep it concise and focused.")

        if content.get("action_verb_score", 0) < 50:
            recommendations.append(
                "Use more action verbs to describe your achievements (e.g., 'achieved', 'managed', 'led').")

        if content.get("quantifiable_score", 0) < 50:
            recommendations.append("Add quantifiable achievements with numbers, percentages, or dollar amounts.")

        # Skills recommendations
        skills = analysis.get("skill_analysis", {})
        if skills.get("diversity_score", 0) < 60:
            recommendations.append(
                "Diversify your skill set across different categories (technical, soft skills, tools).")

        # ATS recommendations
        ats = analysis.get("ats_compatibility", {})
        if ats.get("ats_score", 0) < 70:
            recommendations.append(
                "Improve ATS compatibility by using standard section headers and consistent formatting.")

        if ats.get("keyword_density", 0) < 5:
            recommendations.append("Include more relevant keywords and skills throughout your resume.")

        return recommendations

    async def identify_skill_gaps(self, resume_data: Dict[str, Any], target_role: str = "") -> Dict[str, Any]:
        """Identify skill gaps for target roles"""
        try:
            current_skills = resume_data.get("skills", {})

            # Define common skill requirements for different roles
            role_requirements = {
                "software engineer": {
                    "programming_languages": ["python", "java", "javascript"],
                    "frameworks_libraries": ["react", "node.js", "spring"],
                    "databases": ["mysql", "postgresql", "mongodb"],
                    "tools_technologies": ["git", "docker", "kubernetes"]
                },
                "data scientist": {
                    "programming_languages": ["python", "r", "sql"],
                    "data_science": ["machine learning", "statistics", "data visualization"],
                    "frameworks_libraries": ["pandas", "numpy", "scikit-learn", "tensorflow"],
                    "tools_technologies": ["jupyter", "git"]
                },
                "product manager": {
                    "soft_skills": ["leadership", "communication", "project management"],
                    "tools_technologies": ["jira", "confluence", "figma"],
                    "data_science": ["data analysis"]
                },
                "devops engineer": {
                    "cloud_platforms": ["aws", "azure", "kubernetes", "docker"],
                    "tools_technologies": ["terraform", "ansible", "jenkins"],
                    "programming_languages": ["python", "bash", "go"]
                }
            }

            # Find matching role requirements
            target_role_lower = target_role.lower()
            required_skills = {}

            for role, requirements in role_requirements.items():
                if role in target_role_lower or any(word in target_role_lower for word in role.split()):
                    required_skills = requirements
                    break

            if not required_skills and target_role:
                # Generic analysis if specific role not found
                required_skills = {
                    "programming_languages": ["python", "javascript"],
                    "soft_skills": ["communication", "teamwork", "problem solving"],
                    "tools_technologies": ["git"]
                }

            # Identify gaps
            skill_gaps = {}
            recommendations = []

            for category, required in required_skills.items():
                current_in_category = set(skill.lower() for skill in current_skills.get(category, []))
                required_set = set(skill.lower() for skill in required)

                missing_skills = required_set - current_in_category
                if missing_skills:
                    skill_gaps[category] = list(missing_skills)
                    recommendations.append(
                        f"Consider learning {category.replace('_', ' ')}: {', '.join(missing_skills)}")

            return {
                "target_role": target_role,
                "skill_gaps": skill_gaps,
                "recommendations": recommendations,
                "current_skills": current_skills,
                "required_skills": required_skills,
                "gap_analysis_score": self._calculate_gap_score(current_skills, required_skills)
            }

        except Exception as e:
            logger.error(f"Error identifying skill gaps: {str(e)}")
            raise

    def _calculate_gap_score(self, current_skills: Dict, required_skills: Dict) -> float:
        """Calculate how well current skills match requirements"""
        total_required = sum(len(skills) for skills in required_skills.values())
        if total_required == 0:
            return 100.0

        matched_skills = 0
        for category, required in required_skills.items():
            current_in_category = set(skill.lower() for skill in current_skills.get(category, []))
            required_set = set(skill.lower() for skill in required)
            matched_skills += len(required_set.intersection(current_in_category))

        return round((matched_skills / total_required) * 100, 1)
