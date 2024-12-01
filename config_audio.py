"""Audio device configuration tool."""
import soundcard as sc
import pyaudio
import numpy as np
import time
import sys
import threading
import json
import keyboard
from pathlib import Path
from typing import Any, Dict, Optional, List, Set
import os

from config.config import AUDIO_DEVICES_CONFIG

class AudioDeviceSelector:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.running = True
        self.threads: List[threading.Thread] = []
        self.lock = threading.Lock()
        self.device_positions: Dict[str, int] = {}
        self.current_line = 0
        
        # Selection state - only for input devices
        self.selected_inputs: Set[int] = set()
        self.last_key_time = 0
        self.key_debounce = 0.3  # seconds
        self.config_saved = False
        self.final_config = None
        
        # Audio settings
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 44100
        self.chunk_size = 1024
        
        # Enable Windows ANSI support
        if sys.platform == 'win32':
            os.system('color')
        
        # Load current configuration
        self.current_config = self.load_current_config()
        
        # Pre-populate selections from current config
        self._load_current_selections()

    def _load_current_selections(self):
        """Load current device selections from config."""
        try:
            if self.current_config["input_devices"]:
                for device in self.current_config["input_devices"]:
                    # We'll match these with actual devices later
                    self.selected_inputs.add(device["index"])
        except Exception as e:
            print(f"Error loading selections: {e}")

    def load_current_config(self) -> Dict:
        """Load current audio device configuration if it exists."""
        if AUDIO_DEVICES_CONFIG.exists():
            try:
                with open(AUDIO_DEVICES_CONFIG, 'r') as f:
                    config = json.load(f)
                    # Ensure record_output exists
                    if 'record_output' not in config:
                        config['record_output'] = True
                    return config
            except Exception as e:
                print(f"Error loading current config: {e}")
        return {"input_devices": [], "output_devices": [], "record_output": True}

    def save_config(self, input_devices: List[dict], output_devices: List[dict]) -> None:
        """Save the selected devices to config file."""
        config = {
            "input_devices": [
                {
                    "index": device["index"],
                    "name": device["name"]
                }
                for device in input_devices
            ],
            "output_devices": [
                {
                    "name": device["name"],
                    "id": device["id"]
                }
                for device in output_devices
            ],
            "record_output": self.current_config.get('record_output', True)
        }
        
        try:
            AUDIO_DEVICES_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIO_DEVICES_CONFIG, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\nConfiguration saved to {AUDIO_DEVICES_CONFIG}")
        except Exception as e:
            print(f"\nError saving configuration: {e}")

    def setup_display(self, input_devices: List[dict], output_devices: List[dict]) -> None:
        """Setup the display for monitoring devices."""
        # Clear screen
        print("\033[2J\033[H")
        
        # Print header
        print("\033[1m")  # Bold
        print("Audio Device Configuration")
        print("=" * 80)
        print("\033[0m")  # Reset formatting
        
        # Show current configuration if it exists
        if self.current_config["input_devices"] or self.current_config["output_devices"]:
            print("\033[1mCurrent Configuration:\033[0m")
            if self.current_config["input_devices"]:
                print("\nConfigured Input Devices:")
                for device in self.current_config["input_devices"]:
                    print(f"- {device['name']}")
            if self.current_config["output_devices"]:
                print("\nConfigured Output Devices:")
                for device in self.current_config["output_devices"]:
                    print(f"- {device['name']}")
                print(f"Output Recording: {'Enabled' if self.current_config.get('record_output', True) else 'Disabled'}")
            print("\n" + "=" * 80 + "\n")
        
        # Print input device section
        if input_devices:
            print("\033[1mInput Devices (Select using number keys):\033[0m")
            self.current_line = 5 + (6 if self.current_config["input_devices"] or self.current_config["output_devices"] else 0)
            for i, device in enumerate(input_devices):
                name = device['name']
                if len(name) > 37:
                    name = name[:34] + "..."
                selected = "✓" if i in self.selected_inputs else " "
                print(f"{i+1:2d}. [{selected}] {name:<37} | {'░' * 30} | ---.- dB")
                self.device_positions[name] = self.current_line
                self.current_line += 1
        
        # Add spacing
        print("")
        self.current_line += 1
        
        # Print output device section
        if output_devices:
            print("\033[1mOutput Devices (Automatically included):\033[0m")
            self.current_line += 1
            for device in output_devices:
                name = device['name']
                if len(name) > 37:
                    name = name[:34] + "..."
                print(f"[✓] {name:<40} | {'░' * 30} | ---.- dB")
                self.device_positions[name] = self.current_line
                self.current_line += 1
        
        # Print instructions
        print("\n" + "=" * 80)
        print("Instructions:")
        print("1. Press number keys to select/deselect input devices")
        print("2. Output devices are automatically included")
        print("3. Press 'o' to toggle output recording")
        print("4. Press 's' to save configuration")
        print("5. Press 'q' to quit without saving")
        print("\nMonitoring audio levels...")
        
        sys.stdout.flush()

    def update_level_meter(self, db: float, source_name: str) -> None:
        """Update the level meter for a device."""
        if source_name not in self.device_positions:
            return
            
        with self.lock:
            meter_length = 30
            if db > -float('inf'):
                # Map db range (-60 to 0) to meter length
                db = max(-60, min(0, db))
                level = int((db + 60) * meter_length / 60)
            else:
                level = 0
                
            # Create meter with color based on level
            if level >= meter_length * 0.8:  # Red for high levels
                meter = f"\033[91m{'█' * level}\033[0m{'░' * (meter_length - level)}"
            elif level >= meter_length * 0.5:  # Yellow for medium levels
                meter = f"\033[93m{'█' * level}\033[0m{'░' * (meter_length - level)}"
            else:  # Green for low levels
                meter = f"\033[92m{'█' * level}\033[0m{'░' * (meter_length - level)}"
            
            # Move cursor to device's line
            line = self.device_positions[source_name]
            print(f"\033[{line};47H{meter} | {db:>6.1f} dB", end='', flush=True)
            
            # Return cursor to bottom
            print(f"\033[{self.current_line + 8};0H", end='', flush=True)

    def handle_key_press(self, e, input_devices: List[dict], output_devices: List[dict]) -> bool:
        """Handle key press events with debouncing."""
        try:
            # Only handle key down events
            if e.event_type != 'down':
                return True
                
            # Implement debouncing
            current_time = time.time()
            if current_time - self.last_key_time < self.key_debounce:
                return True
            self.last_key_time = current_time
            
            if e.name == 's':
                # Save configuration
                selected_inputs = [input_devices[i] for i in self.selected_inputs]
                self.save_config(selected_inputs, output_devices)
                self.config_saved = True
                self.final_config = {
                    "input_devices": selected_inputs,
                    "output_devices": output_devices,
                    "record_output": self.current_config.get('record_output', True)
                }
                return False
            elif e.name == 'q':
                return False
            elif e.name == 'o':
                # Toggle output recording
                self.current_config['record_output'] = not self.current_config.get('record_output', True)
                # Refresh display
                self.setup_display(input_devices, output_devices)
            elif e.name.isdigit():
                num = int(e.name) - 1
                if 0 <= num < len(input_devices):
                    # Toggle input device selection
                    if num in self.selected_inputs:
                        self.selected_inputs.remove(num)
                    else:
                        self.selected_inputs.add(num)
                    # Refresh display
                    self.setup_display(input_devices, output_devices)
        except Exception as e:
            print(f"\nError handling key press: {e}")
        return True

    def audio_callback(self, in_data, frame_count, time_info, status, device_name):
        """Process audio data from input device."""
        try:
            if status:
                return (None, pyaudio.paContinue)
            
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = -float('inf')
                
            self.update_level_meter(db, device_name)
            
        except Exception:
            pass
            
        return (None, pyaudio.paContinue)

    def monitor_output_device(self, device: dict) -> None:
        """Monitor a single output device."""
        try:
            mic = sc.get_microphone(id=str(device['id']), include_loopback=True)
            
            while self.running:
                try:
                    data = mic.record(numframes=self.chunk_size, samplerate=self.rate)
                    
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    
                    rms = np.sqrt(np.mean(np.square(data)))
                    
                    if rms > 0:
                        db = 20 * np.log10(rms)
                    else:
                        db = -float('inf')
                    
                    self.update_level_meter(db, device['name'])
                    
                except Exception:
                    pass
                    
                time.sleep(0.05)
                
        except Exception:
            pass

    def get_working_input_devices(self) -> List[dict]:
        """Get list of working input devices."""
        devices = []
        seen_names = set()
        
        for i in range(self.pa.get_device_count()):
            try:
                device_info = self.pa.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # Only input devices
                    name = device_info['name']
                    if name not in seen_names:
                        seen_names.add(name)
                        # Test if device can be opened
                        try:
                            stream = self.pa.open(
                                format=self.format,
                                channels=self.channels,
                                rate=self.rate,
                                input=True,
                                input_device_index=i,
                                frames_per_buffer=self.chunk_size,
                                stream_callback=lambda *args: self.audio_callback(*args, name),
                                start=False
                            )
                            devices.append({
                                "index": i,
                                "name": name,
                                "channels": device_info['maxInputChannels']
                            })
                            stream.close()
                        except Exception:
                            continue
            except Exception:
                continue
        return devices

    def get_working_output_devices(self) -> List[dict]:
        """Get list of working output devices."""
        devices = []
        seen_names = set()
        
        for speaker in sc.all_speakers():
            try:
                if speaker.name not in seen_names:
                    seen_names.add(speaker.name)
                    # Test if device can be opened
                    try:
                        with sc.get_microphone(id=str(speaker.name), include_loopback=True):
                            devices.append({
                                "name": speaker.name,
                                "id": speaker.name
                            })
                    except Exception:
                        continue
            except Exception:
                continue
        return devices

    def monitor_input_device(self, device_info: dict) -> None:
        """Monitor a single input device."""
        try:
            stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=device_info['index'],
                frames_per_buffer=self.chunk_size,
                stream_callback=lambda *args: self.audio_callback(*args, device_info['name'])
            )
            
            while self.running and stream.is_active():
                time.sleep(0.1)
                
            stream.close()
            
        except Exception:
            pass

    def show_final_config(self):
        """Show the final configuration after clearing the screen."""
        # Clear screen
        print("\033[2J\033[H")
        
        if not self.config_saved:
            print("Configuration was not saved.")
            return
            
        print("\033[1mSelected Audio Devices:\033[0m\n")
        
        if self.final_config["input_devices"]:
            print("Selected Input Devices:")
            for device in self.final_config["input_devices"]:
                print(f"- {device['name']}")
        else:
            print("No input devices selected")
            
        print("\nOutput Devices (All included):")
        for device in self.final_config["output_devices"]:
            print(f"- {device['name']}")
            
        print(f"\nOutput Recording: {'Enabled' if self.final_config.get('record_output', True) else 'Disabled'}")
        print("\nConfiguration has been saved and will be used for recording.")

    def monitor_all_devices(self) -> None:
        """Monitor all available audio devices simultaneously."""
        # Get working devices
        input_devices = self.get_working_input_devices()
        output_devices = self.get_working_output_devices()
        
        if not input_devices and not output_devices:
            print("No working audio devices found!")
            return
        
        # Setup display
        self.setup_display(input_devices, output_devices)
        
        # Start monitoring input devices
        for device in input_devices:
            thread = threading.Thread(
                target=self.monitor_input_device,
                args=(device,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Start monitoring output devices
        for device in output_devices:
            thread = threading.Thread(
                target=self.monitor_output_device,
                args=(device,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Handle keyboard input
        keyboard.hook(lambda e: self.handle_key_press(e, input_devices, output_devices))
        
        try:
            # Keep running until 'q' is pressed or 's' saves config
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    def cleanup(self) -> None:
        """Clean up resources and show final configuration."""
        self.running = False
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
            
        # Terminate PyAudio
        self.pa.terminate()
        
        # Show final configuration
        self.show_final_config()

def main():
    """Run the audio device configuration tool."""
    print("Audio Device Configuration Tool")
    print("This tool helps you select which audio devices to record from.")
    print("It will show audio levels to help identify active devices.")
    input("\nPress Enter to continue...")
    
    selector = AudioDeviceSelector()
    
    try:
        selector.monitor_all_devices()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        selector.cleanup()

if __name__ == "__main__":
    main()
