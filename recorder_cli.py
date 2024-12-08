"""CLI script for recording audio from multiple sources to WAV files."""
import sys
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import wave
import fire
from datetime import datetime
import re
import os
import yaml

from config.config import AUDIO_CONFIG
from src.core.storage.metadata_manager import UnifiedMetadataManager
from src.core.audio.recorder import MultiSourceRecorder
from src.core.utils.logging_config import setup_logging

# Enable ANSI color support on Windows
if sys.platform == 'win32':
    os.system('color')
    import msvcrt
else:
    import select

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/recording_debug.log')
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

class AudioRecorderCLI:
    """CLI interface for recording audio from multiple sources."""
    
    def __init__(
        self,
        title: str,
        project_id: Optional[str] = None,
        meeting_id: Optional[str] = None,
        output_alias: str = "Team",
        input_alias: str = "Me",
        save_interval: int = 30
    ):
        """Initialize CLI interface.
        
        Args:
            title: Recording session title
            project_id: Optional project ID (uses default project if not provided)
            meeting_id: Optional meeting ID to continue recording
            output_alias: Alias for output audio sources (default: "Team")
            input_alias: Alias for input audio sources (default: "Me")
            save_interval: How often to save WAV files in seconds (default: 30)
        """
        self.title = title
        self.project_id = project_id
        self.meeting_id = meeting_id
        self.output_alias = output_alias
        self.input_alias = input_alias
        self.save_interval = save_interval
        
        # Initialize components
        self.metadata_manager = UnifiedMetadataManager()
        
        try:
            # Get project configuration
            self.project = self.metadata_manager.get_project(project_id)
            
            # Set up meeting directory
            if meeting_id:
                self.meeting_dir = self.metadata_manager.get_meeting_dir(self.project["key"], meeting_id)
                if not self.meeting_dir.exists():
                    raise ValueError(f"Meeting not found: {meeting_id}")
            else:
                self.meeting_dir = None
            
            # Initialize recorder
            logger.debug("Initializing MultiSourceRecorder")
            self.recorder = MultiSourceRecorder()
            logger.info("MultiSourceRecorder initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            print(f"{Colors.FAIL}Initialization failed: {e}{Colors.ENDC}")
            sys.exit(1)
        
        # State tracking
        self.is_recording = False
        self.should_exit = False
        self.current_session_id = meeting_id
        self.recording_count = 0
        self.last_save_time = 0
        
        # Load device configuration
        self.device_map = {}
        try:
            device_config = AUDIO_CONFIG["devices"]
            for device in device_config.get("input_devices", []):
                self.device_map[device["name"]] = self.input_alias
            for device in device_config.get("output_devices", []):
                self.device_map[device["name"]] = self.output_alias
        except Exception as e:
            logger.error(f"Failed to load device configuration: {e}")
    
    def _create_session(self) -> str:
        """Create a new recording session and return its ID."""
        if not self.current_session_id:
            date_str = datetime.now().strftime("%Y%m%d")
            clean_title = re.sub(r'[^\w\s-]', '', self.title).strip().replace(' ', '_')
            self.current_session_id = f"{date_str}_{clean_title}"
        
        # Create meeting directory
        self.meeting_dir = self.metadata_manager.get_meeting_dir(self.project["key"], self.current_session_id)
        self.meeting_dir.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        metadata = {
            "title": self.title,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_alias": self.output_alias,
            "input_alias": self.input_alias,
            "save_interval": self.save_interval,
            "status": "recording"
        }
        self.metadata_manager.update_meeting_metadata(
            self.project["key"],
            self.current_session_id,
            metadata
        )
        
        return self.current_session_id
    
    def _save_recordings(self, continuous: bool = True):
        """Save current recordings to WAV files.
        
        Args:
            continuous: If True, append to existing file during intervals.
                      If False, create new indexed file for new recording.
        """
        try:
            session_dir = self.meeting_dir / self.current_session_id
            
            # Save recordings with aliased filenames
            saved_files = self.recorder.save_recordings(
                meeting_dir=session_dir,
                session_title=self.title,
                continuous=continuous
            )
            
            # Print status for each saved file
            for device_id, filepath in saved_files.items():
                alias = self._get_device_alias(device_id)
                print(f"{Colors.OKGREEN}Saved recording from {alias} to {filepath.name}{Colors.ENDC}")
            
            # Update last save time
            self.last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to save recordings: {e}", exc_info=True)
            print(f"{Colors.FAIL}Failed to save recordings: {e}{Colors.ENDC}")
    
    def _check_save_interval(self):
        """Check if it's time to save recordings based on save_interval."""
        if self.is_recording and time.time() - self.last_save_time >= self.save_interval:
            logger.info("Save interval reached, saving recordings...")
            self._save_recordings(continuous=True)  # Continuous save during intervals
    
    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.is_recording:
            # Start recording
            if not self.current_session_id:
                self.current_session_id = self._create_session()
            
            logger.info("Starting recording...")
            print(f"\n{Colors.OKGREEN}Starting recording...{Colors.ENDC}")
            
            try:
                # Start recording from all sources
                session_dir = self.meeting_dir / self.current_session_id
                self.recorder.start_recording(session_dir)
                self.is_recording = True
                self.recording_count += 1
                self.last_save_time = time.time()
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
                # Stop recording and save final files
                self.recorder.stop_recording()
                self._save_recordings(continuous=False)  # Create new file for next recording
                
                # Validate recordings
                validation_results = self.recorder.validate_recordings()
                for source_id, is_valid in validation_results.items():
                    alias = self._get_device_alias(source_id)
                    if is_valid:
                        logger.info(f"Recording from {source_id} validated successfully")
                        print(f"{Colors.OKGREEN}Recording from {alias} validated successfully{Colors.ENDC}")
                    else:
                        logger.warning(f"Recording from {source_id} failed validation")
                        print(f"{Colors.WARNING}Recording from {alias} failed validation{Colors.ENDC}")
                
                print(f"{Colors.OKGREEN}Recording stopped. Press Enter to start a new recording in the same session.{Colors.ENDC}")
                
            except Exception as e:
                logger.error(f"Error stopping recording: {e}", exc_info=True)
                print(f"{Colors.FAIL}Error stopping recording: {e}{Colors.ENDC}")
    
    def _check_for_input(self) -> bool:
        """Check for keyboard input in a platform-independent way."""
        if sys.platform == 'win32':
            # Windows-specific input handling
            if msvcrt.kbhit():
                char = msvcrt.getch()
                # Check for Enter key (13 is CR, 10 is LF)
                return char in [b'\r', b'\n']
            return False
        else:
            # Unix-like systems input handling
            rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
            if rlist:
                sys.stdin.readline()
                return True
            return False
    
    def _get_device_alias(self, device_id: str) -> str:
        """Get the alias for a device ID."""
        return self.device_map.get(device_id, device_id)
    
    def run(self):
        """Run the CLI interface."""
        logger.info("Starting CLI interface")
        print(f"""{Colors.BOLD}Audio Recording CLI{Colors.ENDC}

Session: {self.title}
Input Source: {self.input_alias}
Output Source: {self.output_alias}
Save Interval: {self.save_interval} seconds

Controls:
- Press Enter to start/stop recording
- Press Ctrl+C to exit

Recordings will be saved in the meetings directory.
Each audio source will be saved as a separate WAV file.
Files are saved every {self.save_interval} seconds and when recording stops.
Multiple recordings can be made within the same session.
""")
        
        try:
            while not self.should_exit:
                try:
                    # Check if we need to save recordings
                    self._check_save_interval()
                    
                    # Check for Enter key press
                    if self._check_for_input():
                        self._toggle_recording()
                    
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.1)
                        
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
                    self._save_recordings()
                except Exception as e:
                    logger.error(f"Error during shutdown: {e}", exc_info=True)
            
            logger.info("Application shutdown complete")
            print(f"{Colors.OKGREEN}Application shutdown complete.{Colors.ENDC}")

def main(
    title: str,
    project_id: Optional[str] = None,
    meeting_id: Optional[str] = None,
    output_alias: str = "Team",
    input_alias: str = "Me",
    save_interval: int = 30
):
    """Run the Audio Recording CLI.
    
    Args:
        title: Recording session title
        project_id: Optional project ID (uses default project if not provided)
        meeting_id: Optional meeting ID to continue recording
        output_alias: Alias for output audio sources (default: "Team")
        input_alias: Alias for input audio sources (default: "Me")
        save_interval: How often to save WAV files in seconds (default: 30)
    """
    cli = AudioRecorderCLI(
        title=title,
        project_id=project_id,
        meeting_id=meeting_id,
        output_alias=output_alias,
        input_alias=input_alias,
        save_interval=save_interval
    )
    cli.run()

if __name__ == "__main__":
    fire.Fire(main)
