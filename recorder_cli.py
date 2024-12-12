import sys
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import subprocess
import fire
from datetime import datetime
from slugify import slugify
import re
import yaml
import shutil

from src.core.storage.metadata_manager import UnifiedMetadataManager

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/recording_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AudioDeviceDetector:
    """Detects available audio devices using platform-specific commands."""
    
    @staticmethod
    def get_device_list() -> Tuple[List[Dict], List[Dict]]:
        """Get lists of input and output devices."""
        logger.info(f"Detecting audio devices on platform: {sys.platform}")
        if sys.platform == 'win32':
            return AudioDeviceDetector._get_windows_devices()
        elif sys.platform == 'darwin':
            return AudioDeviceDetector._get_macos_devices()
        else:  # Linux
            return AudioDeviceDetector._get_linux_devices()

    @staticmethod
    def _get_windows_devices() -> Tuple[List[Dict], List[Dict]]:
        """Get Windows audio devices using dshow."""
        logger.info("Detecting Windows audio devices using dshow")
        cmd = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.debug(f"FFmpeg command output:\n{result.stderr}")
            lines = result.stderr.split('\n')
            
            inputs = []
            outputs = []
            current_list = None
            in_audio_devices = False
            
            for line in lines:
                if 'DirectShow audio devices' in line:
                    in_audio_devices = True
                    current_list = inputs
                    continue
                elif in_audio_devices and ']  "' in line:
                    # Extract device name between quotes
                    device_name = line.split('"')[1]
                    # Extract alternative name if present
                    alt_name = None
                    if "Alternative name" in line:
                        alt_name = line.split('"')[3]
                    
                    device = {
                        'name': device_name,
                        'id': alt_name if alt_name else device_name,
                        'interface': 'dshow'
                    }
                    current_list.append(device)
            
            # For Windows, outputs are also inputs
            outputs = inputs.copy()
            
            logger.info(f"Detected {len(inputs)} input devices and {len(outputs)} output devices")
            return inputs, outputs
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list Windows devices: {e}")
            return [], []

    @staticmethod
    def _get_macos_devices() -> Tuple[List[Dict], List[Dict]]:
        """Get macOS audio devices using avfoundation."""
        logger.info("Detecting macOS audio devices using avfoundation")
        cmd = ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '""']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.debug(f"FFmpeg command output:\n{result.stderr}")
            lines = result.stderr.split('\n')
            
            inputs = []
            outputs = []
            current_list = None
            
            for line in lines:
                if '[AVFoundation input device]' in line:
                    if 'input devices' in line:
                        current_list = inputs 
                    elif 'output devices' in line:
                        current_list = outputs
                elif current_list is not None and '] ' in line:
                    try:
                        idx = int(line[line.find('[')+1:line.find(']')])
                        name = line[line.find(']')+2:].strip()
                        current_list.append({
                            'name': name,
                            'id': str(idx),
                            'interface': 'avfoundation'
                        })
                    except ValueError:
                        continue
            
            logger.info(f"Detected {len(inputs)} input devices and {len(outputs)} output devices")
            return inputs, outputs
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list macOS devices: {e}")
            return [], []

    @staticmethod
    def _get_linux_devices() -> Tuple[List[Dict], List[Dict]]:
        """Get Linux audio devices using pactl."""
        logger.info("Detecting Linux audio devices using pactl")
        cmd = ['pactl', 'list', 'short', 'sources']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.debug(f"pactl command output for sources:\n{result.stdout}")
            
            inputs = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        inputs.append({
                            'name': parts[1],
                            'id': parts[0],
                            'interface': 'pulse'
                        })
            
            cmd = ['pactl', 'list', 'short', 'sinks']
            result = subprocess.run(cmd, capture_output=True, text=True)
            logger.debug(f"pactl command output for sinks:\n{result.stdout}")
            
            outputs = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        outputs.append({
                            'name': parts[1],
                            'id': parts[0],
                            'interface': 'pulse'
                        })
            
            logger.info(f"Detected {len(inputs)} input devices and {len(outputs)} output devices")
            return inputs, outputs
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list Linux devices: {e}")
            logger.debug(f"pactl command error output:\n{e.stderr}")
            return [], []

class AudioRecorder:
    """Handles audio recording using FFmpeg."""
    
    def __init__(self, output_dir: Path, session_file: Path):
        self.output_dir = output_dir
        self.session_file = session_file
        self.recording = False
        self.current_process: Optional[subprocess.Popen] = None
        
        logger.info("Initializing AudioRecorder")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Session file: {session_file}")
        
        # Check if FFmpeg is installed
        if not shutil.which('ffmpeg'):
            logger.error("FFmpeg is not installed or not in the system PATH")
            raise RuntimeError("FFmpeg is required but not found")

        # Get available devices
        self.input_devices, self.output_devices = AudioDeviceDetector.get_device_list()
        logger.info(f"Detected {len(self.input_devices)} input devices and {len(self.output_devices)} output devices")
        
        if not self.input_devices:
            logger.error("No audio devices detected")
            raise RuntimeError("No audio devices detected")
            
        # Select default devices
        self.input_device = self.input_devices[0]
        self.output_device = self.output_devices[0]
        logger.info(f"Selected input device: {self.input_device['name']}")
        logger.info(f"Selected output device: {self.output_device['name']}")
        
    def _build_ffmpeg_command(self) -> List[str]:
        """Build FFmpeg command based on platform and selected devices."""
        base_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
        ]
        
        if sys.platform == 'linux':
            # For Linux (PulseAudio)
            input_args = [
                '-f', 'pulse', '-i', 'default',  # Microphone input
                '-f', 'pulse', '-i', 'default.monitor',  # System audio output
            ]
        elif sys.platform == 'darwin':
            # For macOS
            input_args = [
                '-f', 'avfoundation',
                '-i', f':{self.input_device["id"]}:{self.output_device["id"]}'
            ]
        else:  # Windows
            # Find Stereo Mix or similar loopback device
            stereo_mix = None
            for device in self.input_devices:
                if any(name.lower() in device['name'].lower() 
                      for name in ['speaker', 'headphone', 'headset', 'output', 'hdmi', 'display audio', 'digital audio', 'stereo mix', 'stereomix', 'wave out', 'what u hear', 'loopback']):
                    stereo_mix = device
                    break
            
            if stereo_mix:
                logger.info(f"Found system audio output device: {stereo_mix['name']}")
                input_args = [
                    '-f', 'dshow',
                    '-i', f'audio={self.input_device["id"]}',  # Microphone
                    '-f', 'dshow',
                    '-i', f'audio={stereo_mix["id"]}'  # System audio
                ]
            else:
                logger.warning("No system audio output device found, recording microphone only")
                input_args = [
                    '-f', 'dshow',
                    '-i', f'audio={self.input_device["id"]}',  # Microphone
                    '-f', 'lavfi',
                    '-i', 'anullsrc'  # Null audio source as placeholder
                ]

        # Output format settings
        output_settings = [
            '-filter_complex', '[0:a][1:a]amerge=inputs=2[aout]',  # Merge both inputs
            '-map', '[aout]',
            '-ac', '2',      # Stereo
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '44100',  # Sample rate
            str(self.session_file)
        ]
        
        return base_cmd + input_args + output_settings

    def _configure_pulseaudio(self):
        """Configure PulseAudio settings for Linux."""
        if sys.platform == 'linux':
            try:
                # Set default source (microphone)
                subprocess.run(['pactl', 'set-default-source', 'default'], check=True)
                # Set default sink (speakers/headphones)
                subprocess.run(['pactl', 'set-default-sink', 'default'], check=True)
                logger.info("PulseAudio configured successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to configure PulseAudio: {e}")

    def start_recording(self) -> bool:
        """Start recording from both input and output devices."""
        if self.recording:
            logger.warning("Recording is already in progress")
            return False
            
        try:
            self._configure_pulseaudio()  # Configure PulseAudio before starting the recording
            cmd = self._build_ffmpeg_command()
            logger.info(f"Starting recording with command: {' '.join(cmd)}")
            
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.recording = True
            
            # Start a thread to log FFmpeg output
            threading.Thread(target=self._log_ffmpeg_output, daemon=True).start()
            
            logger.info("Recording started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def _log_ffmpeg_output(self):
        """Log FFmpeg output for debugging purposes."""
        while self.current_process:
            output = self.current_process.stderr.readline()
            if output:
                logger.debug(f"FFmpeg: {output.strip()}")
            elif self.current_process.poll() is not None:
                break

    def stop_recording(self) -> bool:
        """Stop the current recording."""
        if not self.recording:
            logger.warning("No recording in progress to stop")
            return False
            
        try:
            if self.current_process:
                self.current_process.terminate()
                self.current_process.wait()
            self.recording = False
            logger.info("Recording stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False

class RecordingSession:
    """Manages a recording session."""
    
    def __init__(self, title: str, project_id: str, metadata_manager: UnifiedMetadataManager):
        """Initialize recording session.
        
        Args:
            title: Title of the recording session
            project_id: Project ID to associate with
            metadata_manager: Metadata manager instance
        """
        self.title = title
        self.project_id = project_id
        self.metadata_manager = metadata_manager
        
        # Generate meeting ID based on current date and slugified title
        self.meeting_id = self.metadata_manager.generate_meeting_id(self.title)
        logger.info(f"Generated meeting ID: {self.meeting_id}")
        
        # Generate meeting directory name
        self.meeting_dir_name = self.meeting_id
        logger.info(f"Initializing RecordingSession with title: {title}")
        
        # Get existing metadata or initialize new metadata
        existing_metadata = self.metadata_manager.get_meeting_metadata(self.project_id, self.meeting_id)
        if existing_metadata:
            logger.info("Found existing meeting metadata")
            self.metadata = existing_metadata
        else:
            logger.info("Initializing new meeting metadata")
            self.metadata = {
                "title": self.title,
                "status": "recording",
                "start_time": datetime.utcnow().isoformat(),
                "recording_files": [],
                "vtt_files": []
            }
            # Initialize meeting metadata
            self.metadata_manager.update_meeting_metadata(
                project_id=self.project_id,
                meeting_id=self.meeting_id,
                metadata=self.metadata
            )
        logger.debug("Meeting metadata initialized or retrieved")
        
        # Get meeting directory
        self.meeting_dir = self.metadata_manager.get_meeting_dir(self.project_id, self.meeting_id)
        logger.debug(f"Meeting directory: {self.meeting_dir}")
        
        # Create meeting directory
        self.meeting_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate recording filename with index
        self.session_file = self._generate_recording_filename()
        logger.debug(f"Recording file: {self.session_file}")
        
        # Initialize recorder
        self.recorder = AudioRecorder(self.meeting_dir, self.session_file)
    
    def _generate_recording_filename(self) -> Path:
        """Generate unique recording filename with index."""
        index = len(self.metadata.get("recording_files", [])) + 1
        filename = f"recording_{index:03d}.wav"
        return self.meeting_dir / filename
    
    def start(self) -> bool:
        """Start the recording session."""
        try:
            logger.info("Starting recording session")
            success = self.recorder.start_recording()
            if success:
                self.metadata_manager.update_meeting_metadata(
                    project_id=self.project_id,
                    meeting_id=self.meeting_id,
                    metadata={"status": "recording"}
                )
            return success
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the recording session."""
        try:
            logger.info("Stopping recording session")
            success = self.recorder.stop_recording()
            if success:
                # Add new recording file to list
                relative_path = str(self.session_file.relative_to(self.metadata_manager.data_dir))
                self.metadata["recording_files"].append(relative_path)
                
                # Update metadata with new recording file and status
                self.metadata_manager.update_meeting_metadata(
                    project_id=self.project_id,
                    meeting_id=self.meeting_id,
                    metadata={
                        "status": "recorded",
                        "recording_files": self.metadata["recording_files"]
                    }
                )
            return success
        except Exception as e:
            logger.error(f"Failed to stop session: {e}")
            return False

class RecorderCLI:
    """Command-line interface for audio recording."""
    
    def __init__(
        self,
        title: str,
        project_id: Optional[str] = None,
    ):
        self.title = title
        self.metadata_manager = UnifiedMetadataManager()
        
        # If no project_id provided, get from metadata manager
        if not project_id:
            try:
                project = self.metadata_manager.get_project()
                project_id = project["key"]
            except ValueError as e:
                logger.error(f"Failed to get default project: {e}")
                raise
        
        self.project_id = project_id
        logger.info(f"Initializing RecorderCLI with title: {title}, project_id: {project_id}")
        
        self.session: Optional[RecordingSession] = None
        
    def _print_welcome_message(self):
        """Print welcome message with instructions."""
        print(f"""Audio Recording CLI

Session: {self.title}
Project: {self.project_id}

- Press Ctrl+C to exit
""")
        
    def run(self):
        """Run the CLI interface."""
        logger.info("Starting RecorderCLI run")
        self._print_welcome_message()
        
        try:
            while True:
                try:
                    if not self.session:
                        logger.info("Starting new recording")
                        print("Starting new recording...")
                        self.session = RecordingSession(
                            title=self.title,
                            project_id=self.project_id,
                            metadata_manager=self.metadata_manager
                        )
                        if self.session.start():
                            print(f"Recording to: {self.session.session_file.name}")
                            print("Press Enter to stop recording or Ctrl+C to exit.")
                            # Wait for user input
                            input()
                            print("Stopping recording...")
                            self.session.stop()
                            self.session = None
                            print("Recording stopped. Press Enter to start a new recording or Ctrl+C to exit.")
                            input()
                        else:
                            logger.error("Failed to start recording")
                            print("Failed to start recording.")
                            self.session = None
                            break
                    
                    # Small sleep to avoid tight loop
                    time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error during recording: {e}")
                    print(f"An error occurred: {e}")
                    if self.session and self.session.recorder.recording:
                        print("Attempting to stop recording...")
                        self.session.stop()
                    self.session = None
                    time.sleep(1)  # Wait a bit before retrying
                        
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, exiting")
            print("\nExiting...")
            if self.session and self.session.recorder.recording:
                print("Stopping recording...")
                self.session.stop()
            print("Done.")

def main(
    title: str,
    project_id: Optional[str] = None,
):
    """
    Start a recording session.
    
    Args:
        title: Recording session title
        project_id: Optional project ID (uses default project if not provided)
    """
    logger.info(f"Starting main function with title: {title}, project_id: {project_id}")
    cli = RecorderCLI(title=title, project_id=project_id)
    cli.run()

if __name__ == "__main__":
    fire.Fire(main)
