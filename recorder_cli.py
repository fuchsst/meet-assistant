"""CLI interface for audio recording and transcription."""
import sys
import threading
import time
import logging
from pathlib import Path
from typing import Dict
import fire
import torch
import whisper
from datetime import datetime
import re
import os
import queue
import numpy as np

from config.config import WHISPER_CONFIG, AUDIO_CONFIG, MEETINGS_DIR, MODELS_DIR
from src.core.storage.file_manager import FileManager
from src.core.audio.recorder import MultiSourceRecorder
from src.core.utils.logging_config import setup_logging

# Enable ANSI color support on Windows
if sys.platform == 'win32':
    os.system('color')

# Set up logging with more verbose output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/transcription_debug.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'   # Magenta
    OKBLUE = '\033[94m'   # Blue 
    OKCYAN = '\033[96m'   # Cyan
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'     # Red
    ENDC = '\033[0m'      # Reset to default
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AudioBuffer:
    """Thread-safe audio buffer for real-time processing."""
    def __init__(self, max_size=500):  # Increased buffer size significantly
        self.buffer = queue.Queue(maxsize=max_size)
        self.accumulated = np.array([], dtype=np.float32)
        self.lock = threading.Lock()
        self.total_samples = 0
        self.min_samples_for_transcription = AUDIO_CONFIG["rate"] * 2  # 2 seconds of audio for better context

    def add(self, audio_data):
        """Add audio data to buffer."""
        try:
            # Ensure audio_data is float32 and properly scaled
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                # Apply aggressive scaling for very quiet audio
                max_val = np.abs(audio_data).max()
                if max_val > 0:  # Avoid division by zero
                    if max_val <= 1e-3:  # Very quiet audio
                        scale_factor = 0.5 / max_val  # Scale to use 50% of range
                        audio_data *= scale_factor
                    elif max_val <= 1e-2:  # Moderately quiet audio
                        scale_factor = 0.3 / max_val  # Scale to use 30% of range
                        audio_data *= scale_factor
                audio_data = np.clip(audio_data, -1.0, 1.0)

            # Apply pre-emphasis filter to enhance speech frequencies
            audio_data = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])

            self.buffer.put(audio_data, block=False)
            self.total_samples += len(audio_data)
            logger.debug(f"Added {len(audio_data)} samples to buffer. Total: {self.total_samples}")
        except queue.Full:
            # If buffer is full, try to process accumulated data
            if self.total_samples >= self.min_samples_for_transcription:
                self.get_accumulated()  # Clear the buffer
            logger.warning("Buffer full, attempting to clear")

    def get_accumulated(self, clear=True):
        """Get accumulated audio data."""
        with self.lock:
            chunks = []
            while not self.buffer.empty():
                try:
                    chunk = self.buffer.get_nowait()
                    chunks.append(chunk)
                except queue.Empty:
                    break
            
            if chunks:
                self.accumulated = np.concatenate([self.accumulated] + chunks)
            
            # Only return if we have enough samples
            if len(self.accumulated) >= self.min_samples_for_transcription:
                # Apply normalization to full accumulated buffer
                result = self.accumulated
                rms = np.sqrt(np.mean(np.square(result)))
                if rms > 0:
                    result = result / rms * 0.5  # Normalize to 50% of maximum
                
                if clear:
                    self.accumulated = np.array([], dtype=np.float32)
                    self.total_samples = 0
                logger.debug(f"Got {len(result)} accumulated samples, RMS: {rms}")
                return result
            else:
                logger.debug(f"Not enough samples yet: {len(self.accumulated)}/{self.min_samples_for_transcription}")
                return None

class AudioTranscriptionCLI:
    """CLI interface for audio recording and transcription."""
    
    def __init__(self, language: str = WHISPER_CONFIG["language"], real_time: bool = True):
        """Initialize CLI interface."""
        logger.info(f"Initializing CLI with language={language}, real_time={real_time}")
        
        self.language = language
        self.real_time = real_time
        
        # Initialize components
        logger.info("Initializing components...")
        self.file_manager = FileManager()
        
        # Initialize Whisper
        logger.info("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.debug(f"Loading Whisper model size: {WHISPER_CONFIG['model_size']}")
            self.whisper = whisper.load_model(
                WHISPER_CONFIG["model_size"],
                device=self.device,
                download_root=str(MODELS_DIR)
            )
            logger.info(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            print(f"{Colors.FAIL}Failed to load Whisper model. Please check your installation.{Colors.ENDC}")
            sys.exit(1)
        
        # Initialize audio buffers
        self.audio_buffers = {}
        
        # Initialize recorder with chunk callback
        try:
            logger.debug("Initializing MultiSourceRecorder")
            self.recorder = MultiSourceRecorder(
                chunk_callback=self._process_audio_chunk if self.real_time else None
            )
            logger.info("MultiSourceRecorder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio recorder: {e}", exc_info=True)
            print(f"{Colors.FAIL}Failed to initialize audio recorder. Please check your audio devices.{Colors.ENDC}")
            sys.exit(1)
        
        # State tracking
        self.is_recording = False
        self.should_exit = False
        self.current_meeting_id = None
        
        # Transcription state
        self.transcription_threads: Dict[str, threading.Thread] = {}
        self.transcription_events: Dict[str, threading.Event] = {}
        self.last_transcription = {}  # Store last transcription per source
        
        logger.info("CLI initialization complete")
        print(f"{Colors.OKGREEN}Initialization complete. Using {Colors.BOLD}{self.device.upper()}{Colors.ENDC}{Colors.OKGREEN} for transcription.{Colors.ENDC}")
        if self.real_time:
            print(f"{Colors.OKGREEN}Real-time transcription is enabled.{Colors.ENDC}")

    def _get_initial_prompt(self, device_id: str) -> str:
        """Get contextual initial prompt for transcription."""
        context = self.last_transcription.get(device_id, "")
        prompt = (
            f"The following is a transcription in {self.language}. "
            "The speaker may pause between sentences. "
            "Maintain proper capitalization and punctuation."
        )
        if context:
            prompt += f" Previous context: {context}"
        return prompt

    def _preprocess_text(self, text: str) -> str:
        """Preprocess transcribed text for better formatting."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove leading ellipses
        if text.startswith("..."):
            text = text[3:].lstrip()
        
        # Ensure first letter is capitalized
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure proper sentence endings
        if text and text[-1].isalnum():
            text += "."
        
        return text

    def _create_session(self) -> str:
        """Create a new recording session and return its ID."""
        session_id = self.file_manager.generate_meeting_id()
        logger.info(f"Creating new session with ID: {session_id}")
        
        self.file_manager.create_meeting_directory(session_id)
        
        # Save initial metadata
        metadata = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "language": self.language,
            "device": self.device,
            "model": WHISPER_CONFIG["model_size"],
            "real_time": self.real_time
        }
        self.file_manager.save_metadata(session_id, metadata)
        
        logger.info(f"Created new session: {session_id}")
        return session_id

    def _save_vtt(self, session_id: str, source_id: str, segments: list) -> None:
        """Save transcription segments as VTT file."""
        try:
            session_dir = MEETINGS_DIR / session_id
            safe_name = "".join(c if c.isalnum() else "_" for c in source_id)
            vtt_file = session_dir / f"transcript_{safe_name}.vtt"
            
            with open(vtt_file, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for segment in segments:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    
                    # Format timestamps as HH:MM:SS.mmm
                    start_time = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{start%60:06.3f}"
                    end_time = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{end%60:06.3f}"
                    
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            
            logger.info(f"Saved VTT transcript for {source_id} to {vtt_file}")
            
        except Exception as e:
            logger.error(f"Failed to save VTT for {source_id}: {e}", exc_info=True)

    def _save_transcription(self, session_id: str, source_id: str, text: str):
        """Save transcription text to file."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"[{timestamp}] [{source_id}] {text}"
            
            # Print in color based on source
            color = Colors.OKGREEN if "input" in source_id.lower() else Colors.OKCYAN
            print(f"{color}{formatted_text}{Colors.ENDC}")
            
            # Save to transcript file
            self.file_manager.save_transcript(
                session_id,
                formatted_text + "\n",
                mode='a'
            )
            logger.debug(f"Saved transcription: {formatted_text}")
            
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}", exc_info=True)

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to float32 in range [-1, 1]."""
        try:
            logger.debug(f"Normalizing audio data: dtype={audio_data.dtype}, shape={audio_data.shape}, range=[{audio_data.min()}, {audio_data.max()}]")
            if audio_data.dtype == np.int16:
                normalized = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.float32:
                normalized = np.clip(audio_data, -1, 1)
            else:
                raise ValueError(f"Unsupported audio dtype: {audio_data.dtype}")
            
            logger.debug(f"Normalized audio range: [{normalized.min()}, {normalized.max()}]")
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}", exc_info=True)
            raise

    def _process_audio_chunk(self, chunk: bytes, device_id: str):
        """Process audio chunk for real-time transcription."""
        try:
            if len(chunk) == 0:
                logger.warning(f"Received empty audio chunk for {device_id}")
                return
                
            logger.debug(f"Processing audio chunk for {device_id}, size: {len(chunk)} bytes")
            
            # Convert bytes to numpy array based on format
            try:
                if len(chunk) % 4 == 0:  # float32
                    audio_data = np.frombuffer(chunk, dtype=np.float32)
                    logger.debug(f"Interpreted as float32: {len(audio_data)} samples")
                else:  # int16
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    logger.debug(f"Interpreted as int16: {len(audio_data)} samples")
            except ValueError as e:
                logger.error(f"Failed to convert audio chunk: {e}", exc_info=True)
                return
            
            # Normalize audio data
            try:
                audio_data = self._normalize_audio(audio_data)
            except Exception as e:
                logger.error(f"Failed to normalize audio: {e}", exc_info=True)
                return
            
            # Add to buffer
            if device_id not in self.audio_buffers:
                logger.debug(f"Creating new buffer for {device_id}")
                self.audio_buffers[device_id] = AudioBuffer()
            self.audio_buffers[device_id].add(audio_data)
            
            # Get accumulated audio if we have enough data
            if self.audio_buffers[device_id].total_samples >= AUDIO_CONFIG["rate"]:
                accumulated = self.audio_buffers[device_id].get_accumulated()
                logger.debug(f"Transcribing {len(accumulated)} samples for {device_id}")
                
                try:
                    # Transcribe audio using the language parameter from CLI
                    logger.debug("Starting Whisper transcription")
                    result = self.whisper.transcribe(
                        accumulated,
                        language=self.language,
                        task="transcribe",
                        fp16=torch.cuda.is_available(),
                        temperature=0.0,  # Use dynamic temperature
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        no_speech_threshold=0.6,
                        condition_on_previous_text=True,
                        initial_prompt=self._get_initial_prompt(device_id),
                        beam_size=5,
                        best_of=5
                    )
                    logger.debug(f"Transcription result: {result}")
                    
                    # Process segments
                    for segment in result["segments"]:
                        text = self._preprocess_text(segment["text"])
                        if text and len(text) > 2:
                            self.last_transcription[device_id] = text
                            self._save_transcription(self.current_meeting_id, device_id, text)
                            
                except Exception as e:
                    logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error processing audio chunk for {device_id}: {e}", exc_info=True)

    def _transcribe_file(self, session_id: str, source_id: str, audio_file: Path):
        """Transcribe complete audio file."""
        try:
            logger.info(f"Starting file transcription for {source_id}: {audio_file}")
            
            # Load audio file
            try:
                logger.debug(f"Loading audio file: {audio_file}")
                audio = whisper.load_audio(str(audio_file))
                logger.debug(f"Loaded audio file: {len(audio)} samples, range=[{audio.min()}, {audio.max()}]")
                
                if np.all(audio == 0):
                    logger.warning(f"Audio file contains only zeros: {audio_file}")
                    return
                
            except Exception as e:
                logger.error(f"Failed to load audio file: {e}", exc_info=True)
                return
            
            # Transcribe audio using the language parameter from CLI
            try:
                logger.debug("Starting Whisper transcription")
                result = self.whisper.transcribe(
                    audio,
                    language=self.language,
                    task="transcribe",
                    fp16=torch.cuda.is_available(),
                    temperature=0.0,  # Use dynamic temperature
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    initial_prompt=self._get_initial_prompt(source_id),
                    beam_size=5,
                    best_of=5
                )
                logger.debug(f"Transcription result: {result}")
            except Exception as e:
                logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
                return
            
            # Process segments
            segments = []
            for segment in result["segments"]:
                text = self._preprocess_text(segment["text"])
                if text and len(text) > 2:
                    self._save_transcription(session_id, source_id, text)
                    
                    # Add to segments for VTT
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text
                    })
            
            # Save VTT file
            self._save_vtt(session_id, source_id, segments)
            
        except Exception as e:
            logger.error(f"Transcription failed for {source_id}: {e}", exc_info=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.is_recording:
            # Start recording
            if not self.current_meeting_id:
                self.current_meeting_id = self._create_session()
            
            logger.info("Starting recording...")
            print(f"\n{Colors.OKGREEN}Starting recording...{Colors.ENDC}")
            
            # Get session directory
            session_dir = MEETINGS_DIR / self.current_meeting_id
            
            try:
                # Start recording from all sources
                self.recorder.start_recording(session_dir)
                self.is_recording = True
                print(f"{Colors.OKGREEN}Recording started. Press Enter to stop recording.{Colors.ENDC}")
            except Exception as e:
                logger.error(f"Failed to start recording: {e}", exc_info=True)
                print(f"{Colors.FAIL}Failed to start recording. Please check your audio devices.{Colors.ENDC}")
            
        else:
            # Stop recording
            logger.info("Stopping recording...")
            print(f"\n{Colors.WARNING}Stopping recording...{Colors.ENDC}")
            
            self.is_recording = False
            
            try:
                # Stop recording and save files
                self.recorder.stop_recording()
                saved_files = self.recorder.save_recordings(MEETINGS_DIR / self.current_meeting_id)
                logger.debug(f"Saved audio files: {saved_files}")
                
                if not self.real_time:
                    print(f"{Colors.OKGREEN}Starting post-recording transcription...{Colors.ENDC}")
                    # Create transcription threads for each file
                    threads = []
                    for source_id, audio_file in saved_files.items():
                        logger.debug(f"Creating transcription thread for {source_id}: {audio_file}")
                        thread = threading.Thread(
                            target=self._transcribe_file,
                            args=(self.current_meeting_id, source_id, audio_file),
                            daemon=True
                        )
                        thread.start()
                        threads.append(thread)
                    
                    # Wait for all transcriptions to complete
                    for thread in threads:
                        thread.join()
                    logger.info("All transcription threads completed")
                
                # Validate recordings
                validation_results = self.recorder.validate_recordings()
                for source_id, is_valid in validation_results.items():
                    if is_valid:
                        logger.info(f"Recording from {source_id} validated successfully")
                        print(f"{Colors.OKGREEN}Recording from {source_id} validated successfully{Colors.ENDC}")
                    else:
                        logger.warning(f"Recording from {source_id} failed validation")
                        print(f"{Colors.WARNING}Recording from {source_id} failed validation{Colors.ENDC}")
                
                print(f"{Colors.OKGREEN}Recording stopped. Press Enter to start a new recording.{Colors.ENDC}")
                
            except Exception as e:
                logger.error(f"Error stopping recording: {e}", exc_info=True)
                print(f"{Colors.FAIL}Error stopping recording: {e}{Colors.ENDC}")

    def run(self):
        """Run the CLI interface."""
        logger.info("Starting CLI interface")
        print(f"""{Colors.BOLD}Audio Recording and Transcription CLI{Colors.ENDC}

Controls:
- Press Enter to start/stop recording
- Press Ctrl+C to exit

Recording will be saved in the meetings directory.
Each audio source will be saved as a separate file with its own transcript.
""")
        
        try:
            while not self.should_exit:
                try:
                    # Wait for Enter key
                    input()
                    self._toggle_recording()
                except EOFError:
                    # Handle Ctrl+D (EOF)
                    break
                
        except KeyboardInterrupt:
            # Handle Ctrl+C
            logger.info("Received exit signal")
            print(f"\n{Colors.WARNING}Exiting...{Colors.ENDC}")
            
            if self.is_recording:
                self.is_recording = False
                try:
                    self.recorder.stop_recording()
                except Exception as e:
                    logger.error(f"Error during shutdown: {e}", exc_info=True)
            
            logger.info("Application shutdown complete")
            print(f"{Colors.OKGREEN}Application shutdown complete.{Colors.ENDC}")

def main(language: str = WHISPER_CONFIG["language"], real_time: bool = True):
    """Run the Audio Recording and Transcription CLI.
    
    Args:
        language: Language for transcription (default: auto)
        real_time: Enable real-time transcription during recording (default: True)
    """
    cli = AudioTranscriptionCLI(language, real_time)
    cli.run()

if __name__ == "__main__":
    fire.Fire(main)
