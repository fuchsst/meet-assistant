"""CLI interface for Meeting Assistant."""
import sys
import threading
import time
import logging
import wave
from pathlib import Path
from typing import Optional, Dict, Any
import keyboard
import fire
import pyaudio
import numpy as np
import torch
import whisper
from datetime import datetime

from config.config import (
    WHISPER_CONFIG, VOICE_CONFIG, AUDIO_CONFIG, MEETINGS_DIR, MODELS_DIR
)
from src.core.storage.file_manager import FileManager
from src.core.ai.analysis_pipeline import AnalysisPipeline
from src.core.utils.logging_config import setup_logging

# Enable ANSI color support on Windows
if sys.platform == 'win32':
    import os
    os.system('color')

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class MeetingAssistantCLI:
    """CLI interface for Meeting Assistant with keyboard shortcuts."""
    
    def __init__(
        self,
        language: str = WHISPER_CONFIG["language"],
        voice: str = VOICE_CONFIG["default_voice"],
        title: str = "Untitled Meeting"
    ):
        """Initialize CLI interface."""
        logger.info(f"Initializing CLI with language={language}, voice={voice}, title={title}")
        
        self.language = language
        self.voice = voice
        self.title = title
        
        # Initialize components
        logger.info("Initializing components...")
        self.file_manager = FileManager()
        self.analysis_pipeline = AnalysisPipeline()
        
        # Initialize Whisper
        logger.info("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper = whisper.load_model(
            WHISPER_CONFIG["model_size"],
            device=self.device,
            download_root=str(MODELS_DIR)
        )
        
        # Audio settings
        self.format = pyaudio.paInt16
        self.channels = AUDIO_CONFIG["channels"]
        self.rate = AUDIO_CONFIG["rate"]
        self.chunk = AUDIO_CONFIG["chunk"]
        self.silence_threshold = AUDIO_CONFIG["silence_threshold"]
        self.silence_duration = AUDIO_CONFIG["silence_duration"]
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # State tracking
        self.is_recording = False
        self.should_exit = False
        self.current_meeting_id = None
        
        # Audio processing state
        self.audio_buffer = []
        self.last_audio_time = time.time()
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.transcription_attempts = 0
        self.successful_transcriptions = 0
        
        # Set up keyboard handlers
        logger.info("Setting up keyboard handlers...")
        keyboard.on_press_key("enter", self._toggle_recording)
        keyboard.on_press_key("q", self._find_and_answer_question)
        keyboard.on_press_key("m", self._generate_minutes)
        keyboard.on_press_key("t", self._extract_tasks)
        keyboard.on_press_key("d", self._identify_decisions)
        keyboard.add_hotkey("ctrl+c", self._handle_exit)
        
        logger.info("CLI initialization complete")

    def _create_meeting(self) -> str:
        """Create a new meeting and return its ID."""
        meeting_id = self.file_manager.generate_meeting_id()
        logger.info(f"Creating new meeting with ID: {meeting_id}")
        
        self.file_manager.create_meeting_directory(meeting_id)
        
        # Save initial metadata
        metadata = {
            "title": self.title,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": self.language,
            "voice": self.voice
        }
        self.file_manager.save_metadata(meeting_id, metadata)
        
        logger.info(f"Created new meeting: {meeting_id}")
        return meeting_id

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio data from the stream."""
        if status:
            logger.debug(f"Stream status: {status}")
        
        if self.is_recording:
            try:
                # Convert to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                # Store raw audio
                self.audio_buffer.append(in_data)
                
                # Process for transcription
                self._process_audio(audio_data)
                
            except Exception as e:
                logger.error(f"Error in audio callback: {str(e)}")
                
        return (in_data, pyaudio.paContinue)

    def _process_audio(self, audio_data: np.ndarray):
        """Process audio data for transcription."""
        try:
            # Convert to float32 and normalize
            float_data = audio_data.astype(np.float32) / 32768.0
            
            # Accumulate audio
            self.accumulated_audio = np.concatenate([self.accumulated_audio, float_data])
            
            # Check if we should attempt transcription
            current_time = time.time()
            is_silence = self._is_silence(audio_data)
            time_since_last_audio = current_time - self.last_audio_time
            
            should_transcribe = False
            if is_silence and time_since_last_audio >= self.silence_duration:
                logger.debug("Transcription triggered by silence")
                should_transcribe = True
            elif len(self.accumulated_audio) >= self.rate * 30:  # 30 seconds max
                logger.debug("Transcription triggered by maximum duration")
                should_transcribe = True
                
            if should_transcribe:
                self._attempt_transcription()
                
            if not is_silence:
                self.last_audio_time = current_time
                
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

    def _is_silence(self, audio_array: np.ndarray) -> bool:
        """Detect if an audio chunk is silence."""
        try:
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            is_silent = rms < self.silence_threshold
            logger.debug(f"Silence detection - RMS: {rms:.2f}, "
                        f"threshold: {self.silence_threshold}, "
                        f"is_silent: {is_silent}")
            return is_silent
        except Exception as e:
            logger.error(f"Silence detection error: {str(e)}")
            return False

    def _attempt_transcription(self):
        """Attempt to transcribe accumulated audio."""
        if len(self.accumulated_audio) == 0:
            return
            
        try:
            self.transcription_attempts += 1
            logger.debug(f"Starting transcription attempt {self.transcription_attempts}")
            
            # Get transcription with stricter parameters
            result = self.whisper.transcribe(
                self.accumulated_audio,
                language=self.language,
                task=WHISPER_CONFIG["task"],
                fp16=torch.cuda.is_available(),
                temperature=0.0,
                compression_ratio_threshold=1.5,  # Decreased to be more strict
                no_speech_threshold=0.8,  # Increased further to be more strict about silence
                condition_on_previous_text=True,
                initial_prompt=None  # Removed prompt to prevent bias
            )
            
            # Extract text
            text = result["text"].strip()
            ignore_words = ["Thank you."]
            if text and len(text) > 2 and text not in ignore_words:  # Only process if text is non-empty and longer than 2 chars and not halucinated


                self.successful_transcriptions += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_text = f"[{timestamp}] {text}"
                
                # Print in green color
                print(f"\033[32m{formatted_text}\033[0m")
                
                # Save transcript
                if self.current_meeting_id:
                    self.file_manager.save_transcript(
                        self.current_meeting_id,
                        formatted_text + "\n",
                        mode='a'
                    )
            
            # Reset accumulator
            self.accumulated_audio = np.array([], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_audio_file(self):
        """Save recorded audio to a WAV file."""
        if not self.current_meeting_id or not self.audio_buffer:
            return
            
        try:
            # Get meeting directory
            meeting_dir = MEETINGS_DIR / self.current_meeting_id
            audio_file = meeting_dir / "audio.wav"
            
            # Save as WAV file
            with wave.open(str(audio_file), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.rate)
                wav_file.writeframes(b''.join(self.audio_buffer))
                
            logger.info(f"Saved audio file: {audio_file}")
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            print(f"\nError saving audio file: {str(e)}")

    def _toggle_recording(self, _):
        """Toggle recording state."""
        if not self.is_recording:
            # Start recording
            if not self.current_meeting_id:
                self.current_meeting_id = self._create_meeting()
            
            logger.info("Starting recording...")
            print("\nStarting recording...")
            
            # Reset state
            self.audio_buffer = []
            self.accumulated_audio = np.array([], dtype=np.float32)
            self.last_audio_time = time.time()
            self.transcription_attempts = 0
            self.successful_transcriptions = 0
            
            # Start audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            
        else:
            # Stop recording
            logger.info("Stopping recording...")
            print("\nStopping recording...")
            
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Final transcription attempt
            self._attempt_transcription()
            
            # Save audio file
            self._save_audio_file()
            
            # Log stats
            logger.info(f"Recording session stats - "
                       f"Transcription attempts: {self.transcription_attempts}, "
                       f"Successful transcriptions: {self.successful_transcriptions}")

    def _find_and_answer_question(self, _):
        """Find and answer the last relevant question."""
        if not self.current_meeting_id:
            logger.warning("No active meeting")
            print("\nNo active meeting.")
            return
            
        try:
            # Ensure we have a transcript
            files = self.file_manager.get_meeting_files(self.current_meeting_id)
            if not files["transcript"].exists():
                logger.warning("No transcript available")
                print("\nNo transcript available yet.")
                return
                
            logger.info("Analyzing questions...")
            result = self.analysis_pipeline.execute_custom_task(
                self.current_meeting_id,
                "Find the last relevant question from the transcript and answer it"
            )
            if result and "custom_task" in result:
                logger.info("Question analysis complete")
                print("\nQuestion Analysis:")
                print(result["custom_task"]["result"])
            else:
                logger.info("No questions found")
                print("\nNo questions found in transcript.")
                
        except Exception as e:
            logger.error(f"Error analyzing questions: {str(e)}")
            print(f"\nError analyzing questions: {str(e)}")

    def _generate_minutes(self, _):
        """Generate meeting minutes."""
        if not self.current_meeting_id:
            logger.warning("No active meeting")
            print("\nNo active meeting.")
            return
            
        try:
            # Ensure we have a transcript
            files = self.file_manager.get_meeting_files(self.current_meeting_id)
            if not files["transcript"].exists():
                logger.warning("No transcript available")
                print("\nNo transcript available yet.")
                return
                
            logger.info("Generating meeting summary...")
            summary = self.analysis_pipeline.generate_summary(self.current_meeting_id)
            logger.info("Summary generation complete")
            
            print("\nMeeting Minutes:")
            print(f"Duration: {summary['duration']} seconds")
            print("\nSummary:")
            print(summary["summary"])
            print("\nKey Points:")
            for point in summary["key_points"]:
                if point.strip():  # Only print non-empty points
                    print(f"- {point}")
                    
        except Exception as e:
            logger.error(f"Error generating minutes: {str(e)}")
            print(f"\nError generating minutes: {str(e)}")

    def _extract_tasks(self, _):
        """Extract action items and tasks."""
        if not self.current_meeting_id:
            logger.warning("No active meeting")
            print("\nNo active meeting.")
            return
            
        try:
            # Ensure we have a transcript
            files = self.file_manager.get_meeting_files(self.current_meeting_id)
            if not files["transcript"].exists():
                logger.warning("No transcript available")
                print("\nNo transcript available yet.")
                return
                
            logger.info("Extracting tasks...")
            tasks = self.analysis_pipeline.extract_tasks(self.current_meeting_id)
            if tasks:
                logger.info(f"Found {len(tasks)} tasks")
                print("\nAction Items:")
                for i, task in enumerate(tasks, 1):
                    print(f"{i}. {task}")
            else:
                logger.info("No tasks found")
                print("\nNo tasks found in transcript.")
                
        except Exception as e:
            logger.error(f"Error extracting tasks: {str(e)}")
            print(f"\nError extracting tasks: {str(e)}")

    def _identify_decisions(self, _):
        """Identify key decisions."""
        if not self.current_meeting_id:
            logger.warning("No active meeting")
            print("\nNo active meeting.")
            return
            
        try:
            # Ensure we have a transcript
            files = self.file_manager.get_meeting_files(self.current_meeting_id)
            if not files["transcript"].exists():
                logger.warning("No transcript available")
                print("\nNo transcript available yet.")
                return
                
            logger.info("Identifying decisions...")
            decisions = self.analysis_pipeline.identify_decisions(self.current_meeting_id)
            if decisions:
                logger.info(f"Found {len(decisions)} decisions")
                print("\nKey Decisions:")
                for i, decision in enumerate(decisions, 1):
                    print(f"{i}. {decision}")
            else:
                logger.info("No decisions found")
                print("\nNo decisions found in transcript.")
                
        except Exception as e:
            logger.error(f"Error identifying decisions: {str(e)}")
            print(f"\nError identifying decisions: {str(e)}")

    def _handle_exit(self):
        """Handle exit request."""
        logger.info("Exiting application...")
        print("\nExiting...")
        
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self._attempt_transcription()
            self._save_audio_file()
            
        if hasattr(self, 'audio'):
            self.audio.terminate()
            
        self.should_exit = True
        logger.info("Application shutdown complete")
        sys.exit(0)

    def run(self):
        """Run the CLI interface."""
        logger.info("Starting CLI interface")
        print(f"""Meeting Assistant CLI
Language: {self.language}
Voice: {self.voice}
Title: {self.title}

Controls:
- Enter: Start/Stop recording
- Q: Find and answer last question
- M: Generate minutes
- T: Extract tasks
- D: Identify decisions
- Ctrl+C: Exit

Recording will be saved in the meetings directory.
""")
        
        try:
            # Keep the main thread alive
            while not self.should_exit:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._handle_exit()

def main(
    language: str = WHISPER_CONFIG["language"],
    voice: str = VOICE_CONFIG["default_voice"],
    title: str = "Untitled Meeting"
):
    """Run the Meeting Assistant CLI.
    
    Args:
        language: Language for transcription (default: auto)
        voice: Voice ID for text-to-speech (default: morgan_freeman)
        title: Meeting title (default: Untitled Meeting)
    """
    cli = MeetingAssistantCLI(language, voice, title)
    cli.run()

if __name__ == "__main__":
    fire.Fire(main)
