import time
from typing import Optional


class VoicePipelineProfiler:
    """Profiles each stage of the voice agent pipeline."""
    
    def __init__(self):
        self.markers = {}
        self.turn_id = 0
    
    def mark(self, label: str):
        """Record a timestamp for a pipeline stage."""
        self.markers[label] = time.time()
    
    def elapsed(self, start_label: str, end_label: str) -> float:
        """Get elapsed time between two markers in milliseconds."""
        if start_label not in self.markers or end_label not in self.markers:
            return 0.0
        return (self.markers[end_label] - self.markers[start_label]) * 1000
    
    def reset(self):
        """Clear all markers for next turn."""
        self.markers = {}
        self.turn_id += 1
    
    def report(self):
        """Print a formatted profiling report to console."""
        if not self.markers:
            return
        
        print(f"\n{'='*60}")
        print(f"TURN #{self.turn_id} PIPELINE PROFILE")
        print(f"{'='*60}")
        
        # Check which markers exist and calculate timings
        timings = []
        
        if 'ws_connect' in self.markers and 'speech_end' in self.markers:
            t = self.elapsed('ws_connect', 'speech_end')
            timings.append(f"  Recording duration:     {t:7.1f} ms")
        
        if 'speech_end' in self.markers and 'stt_end' in self.markers:
            t = self.elapsed('speech_end', 'stt_end')
            timings.append(f"  STT processing:         {t:7.1f} ms")
        
        if 'stt_end' in self.markers and 'llm_first_token' in self.markers:
            t = self.elapsed('stt_end', 'llm_first_token')
            timings.append(f"  LLM time-to-first-token:{t:7.1f} ms")
        
        if 'llm_first_token' in self.markers and 'llm_end' in self.markers:
            t = self.elapsed('llm_first_token', 'llm_end')
            timings.append(f"  LLM streaming duration: {t:7.1f} ms")
        
        if 'speech_end' in self.markers and 'llm_end' in self.markers:
            t = self.elapsed('speech_end', 'llm_end')
            timings.append(f"  Total text generation:  {t:7.1f} ms")
        
        if 'tts_first_sentence' in self.markers:
            if 'speech_end' in self.markers:
                t = self.elapsed('speech_end', 'tts_first_sentence')
                timings.append(f"  Time-to-first-audio:    {t:7.1f} ms")
        
        if 'tts_first_sentence' in self.markers and 'tts_last_sentence' in self.markers:
            t = self.elapsed('tts_first_sentence', 'tts_last_sentence')
            timings.append(f"  TTS total duration:     {t:7.1f} ms")
        
        if 'speech_end' in self.markers and 'tts_last_sentence' in self.markers:
            t = self.elapsed('speech_end', 'tts_last_sentence')
            timings.append(f"  TOTAL PIPELINE:         {t:7.1f} ms")
        
        for line in timings:
            print(line)
        
        print(f"{'='*60}\n")


# Global profiler instance
profiler = VoicePipelineProfiler()
