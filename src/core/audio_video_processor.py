"""
Audio and video processing module for CourseMate RAG Application.
Handles audio extraction, transcription, and video processing.
"""

import os
import hashlib
import time
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import concurrent.futures

from langchain_core.documents import Document

from src.config.settings import settings


class AudioVideoProcessor:
    """Handles audio and video processing including transcription and metadata extraction."""
    
    def __init__(self):
        self.data_dir = Path(settings.processing.data_dir)
        self.audio_dir = Path(settings.processing.audio_dir)
        self.transcript_dir = Path(settings.processing.transcript_dir)
        
        # Ensure directories exist
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_data = []
        self.video_data = []
        self.transcript_data = []
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported audio and video file formats."""
        return {
            "audio": [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"],
            "video": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
        }
    
    def is_audio_file(self, file_path: Path) -> bool:
        """Check if file is an audio file."""
        supported_formats = self.get_supported_formats()["audio"]
        return file_path.suffix.lower() in supported_formats
    
    def is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video file."""
        supported_formats = self.get_supported_formats()["video"]
        return file_path.suffix.lower() in supported_formats
    
    def extract_audio_from_video(self, video_path: Path) -> Optional[Path]:
        """Extract audio from video file using ffmpeg."""
        try:
            import subprocess
            
            audio_filename = f"{video_path.stem}_audio.wav"
            audio_path = self.audio_dir / audio_filename
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", str(video_path), 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", "44100", "-ac", "2", 
                str(audio_path), "-y"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and audio_path.exists():
                print(f"Extracted audio: {audio_filename}")
                return audio_path
            else:
                print(f"Failed to extract audio from {video_path.name}: {result.stderr}")
                return None
                
        except ImportError:
            print("ffmpeg not available. Install ffmpeg to extract audio from videos.")
            return None
        except Exception as e:
            print(f"Error extracting audio from {video_path.name}: {e}")
            return None
    
    def transcribe_audio(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio file using Whisper or other speech-to-text service."""
        try:
            import whisper
            
            print(f"Transcribing {audio_path.name}...")
            start_time = time.time()
            
            # Load Whisper model
            model = whisper.load_model("base")
            
            # Transcribe audio
            result = model.transcribe(str(audio_path))
            transcript = result["text"].strip()
            
            print(f"Transcription completed in {time.time() - start_time:.2f} seconds")
            return transcript
            
        except ImportError:
            print("Whisper not available. Install openai-whisper to transcribe audio.")
            return None
        except Exception as e:
            print(f"Error transcribing {audio_path.name}: {e}")
            return None
    
    def process_audio_files(self) -> List[Document]:
        """Process all audio files in the data directory."""
        audio_docs = []
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and self.is_audio_file(file_path):
                print(f"\nProcessing audio file: {file_path.name}")
                
                # Transcribe audio
                transcript = self.transcribe_audio(file_path)
                
                if transcript:
                    # Create document
                    doc = Document(
                        page_content=transcript,
                        metadata={
                            "source_file": file_path.name,
                            "type": "audio",
                            "file_size": file_path.stat().st_size,
                            "duration": self._get_audio_duration(file_path)
                        }
                    )
                    audio_docs.append(doc)
                    
                    # Save transcript
                    self._save_transcript(file_path.stem, transcript)
        
        return audio_docs
    
    def process_video_files(self) -> List[Document]:
        """Process all video files in the data directory."""
        video_docs = []
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and self.is_video_file(file_path):
                print(f"\nProcessing video file: {file_path.name}")
                
                # Extract audio from video
                audio_path = self.extract_audio_from_video(file_path)
                
                if audio_path:
                    # Transcribe extracted audio
                    transcript = self.transcribe_audio(audio_path)
                    
                    if transcript:
                        # Create document
                        doc = Document(
                            page_content=transcript,
                            metadata={
                                "source_file": file_path.name,
                                "type": "video",
                                "file_size": file_path.stat().st_size,
                                "duration": self._get_video_duration(file_path),
                                "extracted_audio": audio_path.name
                            }
                        )
                        video_docs.append(doc)
                        
                        # Save transcript
                        self._save_transcript(file_path.stem, transcript)
        
        return video_docs
    
    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get duration of audio file in seconds."""
        try:
            import librosa
            duration = librosa.get_duration(path=str(audio_path))
            return duration
        except ImportError:
            return None
        except Exception:
            return None
    
    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get duration of video file in seconds."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else None
            cap.release()
            return duration
        except ImportError:
            return None
        except Exception:
            return None
    
    def _save_transcript(self, filename: str, transcript: str) -> None:
        """Save transcript to file."""
        transcript_path = self.transcript_dir / f"{filename}_transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"Saved transcript: {transcript_path.name}")
    
    def get_processed_files(self) -> Tuple[List[Document], List[Document]]:
        """Get all processed audio and video documents."""
        audio_docs = self.process_audio_files()
        video_docs = self.process_video_files()
        
        return audio_docs, video_docs
    
    def clear_cache(self) -> None:
        """Clear all cached audio and transcript files."""
        import shutil
        
        if self.audio_dir.exists():
            shutil.rmtree(self.audio_dir)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        if self.transcript_dir.exists():
            shutil.rmtree(self.transcript_dir)
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        print("Cleared audio and transcript caches.")
