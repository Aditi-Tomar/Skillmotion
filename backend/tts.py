import asyncio
import os
import hashlib
import logging
from typing import Optional
import edge_tts
import aiofiles
from utils import async_retry, cache, run_in_thread

logger = logging.getLogger(__name__)


class EdgeTTSService:
    """Service for text-to-speech using Microsoft Edge TTS (free)"""

    def __init__(self, output_dir: str = "static/audio"):
        self.output_dir = output_dir
        self.voice = "en-US-AriaNeural"  # Default voice
        self.rate = "+0%"  # Normal speed
        self.volume = "+0%"  # Normal volume

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Voice options (all free with Edge TTS)
        self.available_voices = {
            "en-US-AriaNeural": "English (US) - Aria (Female)",
            "en-US-JennyNeural": "English (US) - Jenny (Female)",
            "en-US-GuyNeural": "English (US) - Guy (Male)",
            "en-US-DavisNeural": "English (US) - Davis (Male)",
            "en-GB-SoniaNeural": "English (UK) - Sonia (Female)",
            "en-GB-RyanNeural": "English (UK) - Ryan (Male)",
            "en-AU-NatashaNeural": "English (Australia) - Natasha (Female)",
            "en-CA-ClaraNeural": "English (Canada) - Clara (Female)"
        }

    @async_retry(max_attempts=3)
    async def generate_speech(self, text: str, user_id: str = "anonymous",
                              voice: Optional[str] = None) -> str:
        """Generate speech from text and return URL to audio file"""
        try:
            # Use default voice if none specified
            selected_voice = voice or self.voice

            # Create cache key based on text and voice
            cache_key = self._create_cache_key(text, selected_voice)

            # Check cache first
            cached_url = cache.get(cache_key)
            if cached_url and os.path.exists(cached_url.replace("/static/", "static/")):
                logger.info(f"Using cached audio for text: {text[:50]}...")
                return cached_url

            # Generate unique filename
            filename = f"{cache_key}.mp3"
            file_path = os.path.join(self.output_dir, filename)

            # Generate speech using Edge TTS
            await self._synthesize_speech(text, selected_voice, file_path)

            # Return URL path
            audio_url = f"/static/audio/{filename}"

            # Cache the result
            cache.set(cache_key, audio_url)

            logger.info(f"Generated speech for user {user_id}: {filename}")
            return audio_url

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise

    async def _synthesize_speech(self, text: str, voice: str, output_path: str):
        """Synthesize speech using Edge TTS"""
        try:
            # Clean text for TTS
            cleaned_text = self._clean_text_for_tts(text)

            # Create TTS communication
            communicate = edge_tts.Communicate(
                text=cleaned_text,
                voice=voice,
                rate=self.rate,
                volume=self.volume
            )

            # Generate and save audio
            async with aiofiles.open(output_path, "wb") as file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        await file.write(chunk["data"])

            logger.info(f"Audio saved to {output_path}")

        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            raise

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS pronunciation"""
        import re

        # Remove or replace problematic characters
        text = re.sub(r'[^\w\s\-\.,!?;:\'"()]', ' ', text)

        # Fix common TTS issues
        replacements = {
            'AI': 'A I',
            'API': 'A P I',
            'HTTP': 'H T T P',
            'JSON': 'J S O N',
            'SQL': 'S Q L',
            'CSS': 'C S S',
            'HTML': 'H T M L',
            'URL': 'U R L',
            'UI': 'U I',
            'UX': 'U X'
        }

        for acronym, pronunciation in replacements.items():
            text = re.sub(rf'\b{acronym}\b', pronunciation, text, flags=re.IGNORECASE)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Ensure text isn't too long (Edge TTS has limits)
        if len(text) > 300:
            text = text[:300] + "..."

        return text

    def _create_cache_key(self, text: str, voice: str) -> str:
        """Create a unique cache key for the text and voice combination"""
        content = f"{text}_{voice}_{self.rate}_{self.volume}"
        return hashlib.md5(content.encode()).hexdigest()

    async def get_available_voices(self) -> dict:
        """Get list of available voices"""
        return self.available_voices

    async def set_voice_parameters(self, voice: str = None, rate: str = None,
                                   volume: str = None):
        """Set voice parameters for TTS"""
        if voice and voice in self.available_voices:
            self.voice = voice

        if rate:
            # Validate rate format (+/-XX%)
            if rate.startswith(('+', '-')) and rate.endswith('%'):
                self.rate = rate

        if volume:
            # Validate volume format (+/-XX%)
            if volume.startswith(('+', '-')) and volume.endswith('%'):
                self.volume = volume

    async def batch_generate_speech(self, texts: list, user_id: str = "anonymous") -> list:
        """Generate speech for multiple texts in batch"""
        from utils import batch_process

        async def generate_single(text):
            return await self.generate_speech(text, user_id)

        return await batch_process(texts, generate_single, batch_size=3)

    async def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files to save disk space"""
        import time

        try:
            current_time = time.time()
            removed_count = 0

            for filename in os.listdir(self.output_dir):
                if filename.endswith('.mp3'):
                    file_path = os.path.join(self.output_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)

                    # Remove files older than max_age_hours
                    if file_age > max_age_hours * 3600:
                        os.remove(file_path)
                        removed_count += 1

            logger.info(f"Cleaned up {removed_count} old audio files")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def get_speech_info(self, text: str) -> dict:
        """Get information about speech without generating it"""
        cleaned_text = self._clean_text_for_tts(text)

        # Estimate duration (rough calculation: ~150 words per minute)
        word_count = len(cleaned_text.split())
        estimated_duration = (word_count / 150) * 60  # in seconds

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "word_count": word_count,
            "estimated_duration_seconds": estimated_duration,
            "voice": self.voice,
            "rate": self.rate,
            "volume": self.volume
        }


# Utility function for testing TTS functionality
async def test_tts():
    """Test function for TTS service"""
    tts = EdgeTTSService()

    test_text = "Hello! This is a test of the Skillmotion AI text-to-speech system. How does it sound?"

    try:
        audio_url = await tts.generate_speech(test_text, "test_user")
        print(f"TTS test successful! Audio URL: {audio_url}")

        # Get speech info
        info = await tts.get_speech_info(test_text)
        print(f"Speech info: {info}")

    except Exception as e:
        print(f"TTS test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_tts())
