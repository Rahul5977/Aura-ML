"""
Audio buffer management for real-time transcription
Handles per-client audio buffering and end-of-speech detection
"""

import asyncio
import time
from typing import Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Manages audio buffer for a single client connection"""
    
    def __init__(self, user_id: str, silence_timeout: float = 1.5):
        """
        Initialize audio buffer for a user.
        
        Args:
            user_id: Unique identifier for the user
            silence_timeout: Seconds of silence before triggering transcription
        """
        self.user_id = user_id
        self.buffer = bytearray()
        self.last_chunk_time = time.time()
        self.silence_timeout = silence_timeout
        self.is_recording = False
        self.total_chunks = 0
        
        logger.info(f"Created audio buffer for user {user_id} with {silence_timeout}s timeout")
    
    def add_chunk(self, chunk: bytes) -> None:
        """
        Add an audio chunk to the buffer.
        
        Args:
            chunk: Audio data chunk in bytes
        """
        self.buffer.extend(chunk)
        self.last_chunk_time = time.time()
        self.total_chunks += 1
        self.is_recording = True
        
        logger.debug(f"User {self.user_id}: Added chunk ({len(chunk)} bytes). Buffer size: {len(self.buffer)} bytes")
    
    def get_buffer(self) -> bytes:
        """
        Get current buffer contents.
        
        Returns:
            Current buffer as bytes
        """
        return bytes(self.buffer)
    
    def clear_buffer(self) -> bytes:
        """
        Get buffer contents and clear it.
        
        Returns:
            Buffer contents before clearing
        """
        data = bytes(self.buffer)
        self.buffer.clear()
        self.total_chunks = 0
        self.is_recording = False
        
        logger.debug(f"User {self.user_id}: Cleared buffer ({len(data)} bytes)")
        return data
    
    def is_silent_timeout(self) -> bool:
        """
        Check if silence timeout has been reached.
        
        Returns:
            True if timeout reached and buffer has data
        """
        if not self.is_recording or len(self.buffer) == 0:
            return False
        
        time_since_last_chunk = time.time() - self.last_chunk_time
        return time_since_last_chunk >= self.silence_timeout
    
    def get_buffer_size(self) -> int:
        """Get current buffer size in bytes"""
        return len(self.buffer)
    
    def has_data(self) -> bool:
        """Check if buffer has data"""
        return len(self.buffer) > 0


class AudioBufferManager:
    """
    Manages audio buffers for multiple clients.
    Handles buffer lifecycle and end-of-speech detection.
    """
    
    def __init__(self, silence_timeout: float = 1.5):
        """
        Initialize audio buffer manager.
        
        Args:
            silence_timeout: Default silence timeout for all buffers
        """
        self.buffers: Dict[str, AudioBuffer] = {}
        self.silence_timeout = silence_timeout
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized AudioBufferManager with {silence_timeout}s timeout")
    
    def create_buffer(self, user_id: str) -> AudioBuffer:
        """
        Create a new buffer for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Created AudioBuffer instance
        """
        if user_id in self.buffers:
            logger.warning(f"Buffer already exists for user {user_id}, returning existing")
            return self.buffers[user_id]
        
        buffer = AudioBuffer(user_id, self.silence_timeout)
        self.buffers[user_id] = buffer
        
        logger.info(f"Created buffer for user {user_id}. Total buffers: {len(self.buffers)}")
        return buffer
    
    def get_buffer(self, user_id: str) -> Optional[AudioBuffer]:
        """
        Get buffer for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            AudioBuffer if exists, None otherwise
        """
        return self.buffers.get(user_id)
    
    def remove_buffer(self, user_id: str) -> None:
        """
        Remove buffer for a user.
        
        Args:
            user_id: User identifier
        """
        if user_id in self.buffers:
            del self.buffers[user_id]
            logger.info(f"Removed buffer for user {user_id}. Remaining buffers: {len(self.buffers)}")
    
    def add_audio_chunk(self, user_id: str, chunk: bytes) -> bool:
        """
        Add audio chunk to user's buffer.
        
        Args:
            user_id: User identifier
            chunk: Audio data chunk
            
        Returns:
            True if chunk was added, False if buffer doesn't exist
        """
        buffer = self.get_buffer(user_id)
        if buffer:
            buffer.add_chunk(chunk)
            return True
        
        logger.warning(f"No buffer found for user {user_id}")
        return False
    
    def check_silence_timeout(self, user_id: str) -> bool:
        """
        Check if user's buffer has reached silence timeout.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if timeout reached, False otherwise
        """
        buffer = self.get_buffer(user_id)
        if buffer:
            return buffer.is_silent_timeout()
        return False
    
    def get_and_clear_buffer(self, user_id: str) -> Optional[bytes]:
        """
        Get buffer contents and clear it.
        
        Args:
            user_id: User identifier
            
        Returns:
            Buffer contents or None if buffer doesn't exist
        """
        buffer = self.get_buffer(user_id)
        if buffer and buffer.has_data():
            return buffer.clear_buffer()
        return None
    
    async def start_monitoring(self, callback):
        """
        Start monitoring all buffers for silence timeout.
        
        Args:
            callback: Async function to call when timeout detected
                     Signature: async def callback(user_id: str, audio_data: bytes)
        """
        self.monitoring_task = asyncio.create_task(self._monitor_buffers(callback))
        logger.info("Started buffer monitoring task")
    
    async def stop_monitoring(self):
        """Stop monitoring buffers"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped buffer monitoring task")
    
    async def _monitor_buffers(self, callback):
        """
        Internal method to monitor buffers for silence timeout.
        
        Args:
            callback: Async function to call when timeout detected
        """
        while True:
            try:
                # Check each buffer
                for user_id, buffer in list(self.buffers.items()):
                    if buffer.is_silent_timeout():
                        audio_data = buffer.clear_buffer()
                        if audio_data:
                            logger.info(f"Silence timeout detected for user {user_id}, triggering transcription")
                            await callback(user_id, audio_data)
                
                # Check every 100ms
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in buffer monitoring: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> dict:
        """
        Get statistics about all buffers.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            "total_buffers": len(self.buffers),
            "buffers": {}
        }
        
        for user_id, buffer in self.buffers.items():
            stats["buffers"][user_id] = {
                "buffer_size": buffer.get_buffer_size(),
                "total_chunks": buffer.total_chunks,
                "is_recording": buffer.is_recording,
                "time_since_last_chunk": time.time() - buffer.last_chunk_time
            }
        
        return stats


# Global audio buffer manager instance
audio_buffer_manager = AudioBufferManager(silence_timeout=1.5)
