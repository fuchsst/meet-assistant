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
            
            for line in lines:
                if 'DirectShow audio devices' in line:
                    current_list = inputs
                elif 'DirectShow video devices' in line:
                    break
                elif current_list is not None and ']  "' in line:
                    device_name = line.split('"')[1]
                    current_list.append({
                        'name': device_name,
                        'id': device_name,
                        'interface': 'dshow'
                    })
            
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
        
        if not self.input_devices or not self.output_devices:
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
            input_args = [
                '-f', 'dshow',
                '-i', f'audio={self.input_device["id"]}',
                '-f', 'dshow',
                '-i', f'audio={self.output_device["id"]}'
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
    
    def __init__(self, title: str, output_dir: Path):
        self.title = title
        self.output_dir = output_dir
        logger.info(f"Initializing RecordingSession with title: {title}")
        logger.debug(f"Output directory: {output_dir}")
        
        self.session_file = self._generate_session_filename()
        logger.info(f"Generated session filename: {self.session_file}")
        
        self.recorder = AudioRecorder(output_dir, self.session_file)
        
    def _generate_session_filename(self) -> Path:
        """Generate unique session filename."""
        date_str = datetime.now().strftime("%Y%m%d")
        title_slug = slugify(self.title)
        
        # Find next available sequence number
        seq = 1
        while True:
            filename = f"{date_str}_{title_slug}_{seq:03d}.wav"
            file_path = self.output_dir / filename
            if not file_path.exists():
                return file_path
            seq += 1
        
    def start(self) -> bool:
        """Start the recording session."""
        try:
            logger.info("Starting recording session")
            return self.recorder.start_recording()
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the recording session."""
        try:
            logger.info("Stopping recording session")
            return self.recorder.stop_recording()
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
        self.project_id = project_id
        logger.info(f"Initializing RecorderCLI with title: {title}, project_id: {project_id}")
        
        self.output_dir = self.get_meeting_dir(project_id)
        logger.info(f"Output directory: {self.output_dir}")
        
        self.session: Optional[RecordingSession] = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {self.output_dir}")
        
    def get_meeting_dir(self, project_id: Optional[str] = None) -> Path:
        """Get the directory for storing meeting recordings."""
        base_dir = Path("data")
        date_str = datetime.now().strftime("%Y%m%d")
        if project_id:
            return base_dir / f"{project_id}/meetings"
        return base_dir / f"{date_str}_default/meetings"
        
    def _print_welcome_message(self):
        """Print welcome message with instructions."""
        print(f"""Audio Recording CLI

Session: {self.title}
Output Directory: {self.output_dir}

- Press Ctrl+C to exit
""")
        
    def run(self):
        """Run the CLI interface."""
        logger.info("Starting RecorderCLI run")
        self._print_welcome_message()
        
        try:
            if not self.session:
                logger.info("Starting new recording")
                print("Starting new recording...")
                self.session = RecordingSession(self.title, self.output_dir)
                if self.session.start():
                    print(f"Recording to: {self.session.session_file.name}")
                    print("Press Enter to stop recording.")
                else:
                    logger.error("Failed to start recording")
                    print("Failed to start recording.")
                    self.session = None
            else:
                logger.info("Stopping current recording")
                print("Stopping recording...")
                self.session.stop()
                self.session = None
                print("Recording stopped. Press Enter to start a new recording.")
                        
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
