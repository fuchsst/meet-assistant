"""Real-time transcription using Whisper with speaker diarization."""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator, Union, Dict, List
import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment
from scipy import signal

from config.config import WHISPER_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)

class WhisperProcessor:
    """Handles real-time transcription using Whisper with speaker diarization."""

    def __init__(self):
        """Initialize Whisper model and settings."""
        self.model_size = WHISPER_CONFIG["model_size"]
        self.language = WHISPER_CONFIG["language"]
        self.task = WHISPER_CONFIG["task"]
        self.model: Optional[WhisperModel] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "float32"
        
        # Initialize models
        self._load_models()
        
        # Track transcription attempts
        self.transcription_attempts = 0
        self.successful_transcriptions = 0
        
        # Maximum audio duration (15 seconds)
        self.max_duration = 15
        
        # Speaker diarization settings
        self.min_speakers = 1
        self.max_speakers = 5
        self.speaker_embeddings = {}
        
        # Audio preprocessing settings
        self._init_audio_processing()
        
        logger.info(f"WhisperProcessor initialized with {self.model_size} model")
        logger.info(f"Language: {self.language}, Task: {self.task}, Device: {self.device}")

    def _init_audio_processing(self):
        """Initialize audio processing components."""
        # Silence detection settings
        self.silence_threshold_db = -20  # dB
        self.silence_chunk_size = 10  # ms
        self.min_silence_duration = 0.1  # seconds
        
        # Initialize filter
        nyquist = 16000 * 0.5  # Assuming 16kHz sample rate
        low_freq = 100.0  # Hz
        high_freq = 7000.0  # Hz
        self.low = low_freq / nyquist
        self.high = high_freq / nyquist
        self.filter_b, self.filter_a = signal.butter(6, [self.low, self.high], btype='band')
        
        # Noise reduction settings
        self.noise_reduction_strength = 0.75
        self.noise_frame_count = 0
        self.min_noise_frames = 10
        self.noise_floor = None
        
        # Compression settings
        self.compression_ratio = 2.0
        self.compression_threshold = 0.3
        
        # Overlap settings
        self.overlap_size = 1600  # 100ms at 16kHz
        self.overlap_buffer = None

    def _load_models(self) -> None:
        """Load or download Whisper and diarization models."""
        try:
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}")
            
            # Create models directory if it doesn't exist
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Load faster-whisper model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(MODELS_DIR)
            )
            
            # Load speaker diarization pipeline
            self.diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            if torch.cuda.is_available():
                self.diarization = self.diarization.to(torch.device(self.device))
            
            # Perform a test transcription
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            _ = self.model.transcribe(test_audio)
            
            logger.info(f"Models loaded successfully on {self.device}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}", exc_info=True)
            raise

    def _trim_silence(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Trim silence from the beginning and end of audio."""
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Calculate RMS energy in small windows
        frame_length = int(sample_rate * self.silence_chunk_size / 1000)
        frames = np.array_split(audio, len(audio) // frame_length)
        frame_energies = [np.sqrt(np.mean(frame**2)) for frame in frames]
        
        # Convert threshold from dB to linear
        threshold = 10**(self.silence_threshold_db/20)
        
        # Find start and end of speech
        start_frame = 0
        end_frame = len(frames) - 1
        
        for i, energy in enumerate(frame_energies):
            if energy > threshold:
                start_frame = max(0, i - 1)  # Keep one frame before speech
                break
        
        for i in range(len(frame_energies) - 1, -1, -1):
            if frame_energies[i] > threshold:
                end_frame = min(len(frames) - 1, i + 1)  # Keep one frame after speech
                break
        
        # Convert frame indices to sample indices
        start_sample = start_frame * frame_length
        end_sample = min(len(audio), (end_frame + 1) * frame_length)
        
        return audio[start_sample:end_sample]

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio with noise reduction, filtering, and normalization."""
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Update noise floor estimate
        if self.noise_frame_count < self.min_noise_frames:
            if self.noise_floor is None:
                self.noise_floor = np.abs(audio)
            else:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * np.abs(audio)
            self.noise_frame_count += 1
        
        # Apply noise reduction
        if self.noise_floor is not None:
            audio = audio - (self.noise_floor * self.noise_reduction_strength)
        
        # Apply bandpass filter
        audio = signal.filtfilt(self.filter_b, self.filter_a, audio)
        
        # Apply compression
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms > self.compression_threshold:
            reduction = 1 + (self.compression_ratio - 1) * (rms - self.compression_threshold)
            audio = audio / reduction
        
        # Handle overlap for smooth transitions
        if self.overlap_buffer is not None:
            fade_in = np.linspace(0, 1, self.overlap_size)
            fade_out = np.linspace(1, 0, self.overlap_size)
            audio[:self.overlap_size] = (
                fade_out * self.overlap_buffer +
                fade_in * audio[:self.overlap_size]
            )
        
        # Save overlap buffer for next chunk
        self.overlap_buffer = audio[-self.overlap_size:].copy()
        
        return audio

    def _process_diarization(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[Segment, str]:
        """Process speaker diarization."""
        try:
            # Get diarization results
            diarization = self.diarization({"waveform": audio_array, "sample_rate": sample_rate})
            
            # Extract speaker segments
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = Segment(turn.start, turn.end)
                speaker_segments[segment] = speaker
            
            return speaker_segments
            
        except Exception as e:
            logger.error(f"Diarization error: {str(e)}")
            return {}

    def transcribe_chunk(self, audio_data: Union[np.ndarray, bytes], language: Optional[str] = None) -> str:
        """Transcribe an audio chunk in real-time."""
        if not self.model:
            logger.error("Whisper model not initialized")
            raise RuntimeError("Whisper model not initialized")

        try:
            self.transcription_attempts += 1
            logger.debug(f"Starting transcription attempt {self.transcription_attempts}")
            
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                logger.debug("Converting audio bytes to numpy array")
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data

            # Ensure audio is float32 and normalized
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()

            # Check audio duration
            audio_duration = len(audio_array) / 16000  # assuming 16kHz sample rate
            if audio_duration > self.max_duration:
                logger.warning(f"Audio chunk too long ({audio_duration:.2f}s), truncating to {self.max_duration}s")
                samples_to_keep = int(16000 * self.max_duration)
                audio_array = audio_array[:samples_to_keep]
            
            # Skip if too short
            if audio_duration < 0.1:  # Less than 100ms
                logger.debug("Audio chunk too short, skipping transcription")
                return ""

            # Trim silence
            audio_array = self._trim_silence(audio_array)
            if len(audio_array) < 1600:  # Less than 100ms after trimming
                logger.debug("Audio chunk silent after trimming")
                return ""

            # Process audio
            audio_array = self._process_audio(audio_array)

            # Log audio stats
            rms = np.sqrt(np.mean(np.square(audio_array)))
            peak = np.abs(audio_array).max()
            logger.debug(
                f"Audio stats - Duration: {audio_duration:.2f}s, "
                f"RMS: {rms:.3f}, Peak: {peak:.3f}"
            )

            # Skip if signal too weak
            if rms < 0.001:
                logger.debug(f"Audio signal too weak (RMS: {rms:.6f})")
                return ""

            # Get transcription
            logger.debug("Starting Whisper transcription")
            segments, info = self.model.transcribe(
                audio_array,
                language=language or self.language,
                task=self.task,
                beam_size=5,
                best_of=5,
                temperature=0.0,  # Use dynamic temperature
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=(
                    f"The following is a transcription in {language or self.language}. "
                    "The speaker may pause between sentences. "
                    "Maintain proper capitalization and punctuation."
                )
            )
            
            # Process speaker diarization if audio is long enough
            speaker_segments = {}
            if len(audio_array) >= 16000:  # Only process if at least 1 second
                speaker_segments = self._process_diarization(audio_array)
            
            # Combine transcription with speaker information
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_segments = []
            
            for segment in segments:
                segment_time = Segment(segment.start, segment.end)
                speaker = "UNKNOWN"
                
                # Find matching speaker segment
                for s, spk in speaker_segments.items():
                    if s.overlaps(segment_time):
                        speaker = spk
                        break
                
                # Clean up text
                text = segment.text.strip()
                if text and len(text) > 2:
                    formatted_text = f"[{timestamp}] <{speaker}> {text}"
                    formatted_segments.append(formatted_text)
            
            if formatted_segments:
                self.successful_transcriptions += 1
                result = "\n".join(formatted_segments)
                logger.info(f"Transcription successful: {result}")
                return result
            else:
                logger.debug("No text detected in transcription result")
                return ""

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            if torch.cuda.is_available():
                logger.error(f"CUDA memory state: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
                torch.cuda.empty_cache()
            return ""

    def transcribe_stream(self, audio_stream: Generator[Union[np.ndarray, bytes], None, None]) -> Generator[str, None, None]:
        """Transcribe an audio stream in real-time."""
        if not self.model:
            logger.error("Whisper model not initialized")
            raise RuntimeError("Whisper model not initialized")

        try:
            logger.info("Starting stream transcription")
            for chunk_num, chunk in enumerate(audio_stream, 1):
                logger.debug(f"Processing stream chunk {chunk_num}")
                transcription = self.transcribe_chunk(chunk)
                if transcription:
                    logger.debug(f"Stream chunk {chunk_num} transcribed: {transcription}")
                    yield transcription
                else:
                    logger.debug(f"No transcription for stream chunk {chunk_num}")

        except Exception as e:
            logger.error(f"Stream transcription error: {str(e)}", exc_info=True)
            yield ""

    def transcribe_file(self, audio_path: Path) -> List[Dict]:
        """Transcribe an entire audio file with detailed segments and speaker diarization."""
        if not self.model:
            logger.error("Whisper model not initialized")
            raise RuntimeError("Whisper model not initialized")

        try:
            logger.info(f"Starting file transcription: {audio_path}")
            
            # Load audio file
            import soundfile as sf
            audio_array, sample_rate = sf.read(str(audio_path))
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # Convert to mono
            
            # Normalize audio
            audio_array = audio_array.astype(np.float32)
            if np.abs(audio_array).max() > 1.0:
                audio_array = audio_array / np.abs(audio_array).max()
            
            # Trim silence
            audio_array = self._trim_silence(audio_array, sample_rate)
            
            # Process audio
            audio_array = self._process_audio(audio_array)
            
            # Get speaker diarization
            speaker_segments = self._process_diarization(audio_array, sample_rate)
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_array,
                language=self.language,
                task=self.task,
                beam_size=5,
                best_of=5,
                temperature=0.0,  # Use dynamic temperature
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=(
                    f"The following is a transcription in {self.language}. "
                    "The speaker may pause between sentences. "
                    "Maintain proper capitalization and punctuation."
                )
            )
            
            # Format segments with timestamps and speaker information
            formatted_segments = []
            for segment in segments:
                segment_time = Segment(segment.start, segment.end)
                speaker = "UNKNOWN"
                
                # Find matching speaker segment
                for s, spk in speaker_segments.items():
                    if s.overlaps(segment_time):
                        speaker = spk
                        break
                
                # Clean up text
                text = segment.text.strip()
                if text and len(text) > 2:
                    formatted_segment = {
                        "start": self._format_timestamp(segment.start),
                        "end": self._format_timestamp(segment.end),
                        "speaker": speaker,
                        "text": text
                    }
                    formatted_segments.append(formatted_segment)
                    logger.debug(f"Processed segment: {formatted_segment}")

            logger.info(f"File transcription completed: {len(formatted_segments)} segments")
            return formatted_segments

        except Exception as e:
            logger.error(f"File transcription error: {str(e)}", exc_info=True)
            raise

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __del__(self):
        """Clean up resources."""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            try:
                before_mem = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                after_mem = torch.cuda.memory_allocated()
                logger.info(f"CUDA memory cleared: {(before_mem - after_mem) / 1024**2:.2f} MB freed")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {str(e)}")
