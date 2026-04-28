# src/voice_commands.py
import pyaudio
import json
from vosk import Model, KaldiRecognizer
import queue
import threading
import time

class VoskCommandListener:
    def __init__(self, model_path="/home/e3b/elec3442-audiovision/elec3442-audiovision-voice-command-integration/vosk-model-small-en-us-0.15", commands=None):
        """
        Initialize the Vosk voice command listener
        
        Args:
            model_path: Path to the Vosk model directory
            commands: Dictionary mapping voice commands to action identifiers
        """
        print(f"Loading Vosk model from {model_path}...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 44100)
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.recognized_commands = []
        self.commands = commands if commands is not None else {}
        self.callback = None
        self.partial_text = ""
        self.detected_actions = []
        
    def set_callback(self, callback):
        """
        Set a callback function to be called when a command is recognized
        
        Args:
            callback: Function that takes (command_action, recognized_text) parameters
        """
        self.callback = callback
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function to receive audio data"""
        if status:
            print(f"Audio error: {status}")
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def start(self):
        """Start listening for voice commands (alias for start_listening)"""
        self.start_listening()
    
    def start_listening(self):
        """Start listening for voice commands"""
        self.is_listening = True
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # List available input devices (for debugging)
        print("\nAvailable audio input devices:")
        input_devices = []
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                input_devices.append(i)
                print(f"  Device {i}: {dev['name']}")
        
        if not input_devices:
            print("WARNING: No input devices found!")
        
        # Find default input device
        default_device = None
        try:
            default_info = self.p.get_default_input_device_info()
            default_device = default_info['index']
            print(f"Using default input device: {default_info['name']}")
        except:
            if input_devices:
                default_device = input_devices[0]
                print(f"No default device, using first available: {default_device}")
        
        # Open audio stream with correct sample rate
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=default_device,
                frames_per_buffer=4000,
                stream_callback=self.audio_callback
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            print("Trying with default device...")
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=4000,
                stream_callback=self.audio_callback
            )
        
        self.stream.start_stream()
        print("\n" + "="*50)
        print("🎤 Voice command system active!")
        print("="*50)
        
        if self.commands:
            print("\nAvailable commands:")
            for cmd in self.commands.keys():
                print(f"  • {cmd}")
        print("\nListening... (Press Ctrl+C to stop)\n")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def poll_command(self):
        """
        Poll for detected commands (non-blocking)
        
        Returns:
            Command action string if a command was detected, None otherwise
        """
        # First check for detected actions from command matching
        action = self.get_detected_action()
        if action:
            return action
        
        # Also check recognized text as fallback
        latest = self.get_latest_command()
        if latest:
            text_lower = latest.lower()
            for command_phrase, command_action in self.commands.items():
                if command_phrase.lower() in text_lower:
                    return command_action
        
        return None
    
    def stop(self):
        """Stop listening (alias for stop_listening)"""
        self.stop_listening()
    
    def stop_listening(self):
        """Stop listening and clean up resources"""
        self.is_listening = False
        
        if hasattr(self, 'stream'):
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            
        if hasattr(self, 'p'):
            self.p.terminate()
            
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
            
        print("\n🔇 Stopped listening for commands")
    
    def _process_audio(self):
        """Process audio from queue and perform speech recognition"""
        while self.is_listening:
            try:
                # Get audio data from queue (with timeout)
                data = self.audio_queue.get(timeout=0.1)
                
                # Process with Vosk
                if self.recognizer.AcceptWaveform(data):
                    # Final result
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text:
                        print(f"\n🎙️ Recognized: {text}")
                        self.recognized_commands.append(text)
                        
                        # Check for command matches
                        self._check_for_commands(text)
                        
                        # Reset partial text
                        self.partial_text = ""
                else:
                    # Partial result (for real-time feedback)
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get('partial', '').strip()
                    
                    if partial_text and partial_text != self.partial_text:
                        self.partial_text = partial_text
                        # Print partial without newline (overwrite)
                        print(f"\r💭 Partial: {partial_text}", end='', flush=True)
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_listening:
                    print(f"\n⚠️ Error processing audio: {e}")
    
    def _check_for_commands(self, text):
        """
        Check if recognized text matches any known commands
        
        Args:
            text: The recognized text string
        """
        if not self.commands:
            return
            
        text_lower = text.lower()
        
        # Check each command pattern
        for command_phrase, command_action in self.commands.items():
            # Check if command phrase appears in recognized text
            if command_phrase.lower() in text_lower:
                print(f"\n✅ COMMAND DETECTED: {command_phrase} -> {command_action}")
                
                # Call callback if registered
                if self.callback:
                    try:
                        self.callback(command_action, text)
                    except Exception as e:
                        print(f"Error in callback: {e}")
                
                # Also store as command action
                self.detected_actions.append(command_action)
                break
    
    def get_latest_command(self):
        """
        Get the most recent recognized command text
        
        Returns:
            String of the most recent command, or None if none available
        """
        if self.recognized_commands:
            return self.recognized_commands.pop(0)
        return None
    
    def get_detected_action(self):
        """
        Get the most recent detected action (from command mapping)
        
        Returns:
            Action string from commands mapping, or None if none available
        """
        if self.detected_actions:
            return self.detected_actions.pop(0)
        return None
    
    def is_active(self):
        """Check if listener is currently active"""
        return self.is_listening and hasattr(self, 'stream') and self.stream.is_active()


# Example standalone test
if __name__ == "__main__":
    # Test commands
    test_commands = {
        "hello": "GREETING",
        "stop": "COMMAND_STOP",
        "go": "COMMAND_GO",
        "navigation": "NAV_MODE",
        "assist": "ASSIST_MODE"
    }
    
    def on_command(action, text):
        print(f"\n🔥 CALLBACK: Action='{action}', Text='{text}'")
    
    # Create listener
    listener = VoskCommandListener(
        model_path="vosk-model-small-en-us-0.15",
        commands=test_commands
    )
    
    listener.set_callback(on_command)
    
    try:
        listener.start()
        print("\nListening for 30 seconds... (or press Ctrl+C to stop early)")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        listener.stop()
