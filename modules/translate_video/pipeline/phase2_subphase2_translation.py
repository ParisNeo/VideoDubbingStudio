"""
Phase 2, Subphase 2: Text Translation

Handles translation of transcribed text using Lollms.
This subphase loads the Lollms client, translates all segments, then unloads
to free VRAM for the TTS subphase.

Design principles:
- Load Lollms on-demand, unload after all translation is done
- Support same-language pass-through (no translation needed)
- Handle speaker removal action (skip translation for removed speakers)
- Progressive updates to UI via callback
"""

import asyncio
import logging
import traceback
from typing import List, Dict, Any, Optional, Callable
import torch
from core.resources import manager
import torch 
from .phase2_models import TranslationSegment

logger = logging.getLogger("phase2_subphase2_translation")


class TranslationSubphase:
    """
    Handles text translation using Lollms.
    VRAM-efficient: loads Lollms, translates all, unloads immediately.
    """
    
    def __init__(
        self,
        task_id: str,
        target_language: str = "en",
        source_language: str = "auto",
        speaker_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        translation_update_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ):
        self.task_id = task_id
        self.target_language = target_language
        self.source_language = source_language
        self.speaker_config = speaker_config or {}
        self.progress_callback = progress_callback
        self.translation_update_callback = translation_update_callback
        
        self._lollms = None
        self._loaded = False
        
        # Build merge map: merged speaker -> master speaker
        self._merge_map: Dict[str, str] = {}
        self._master_speakers: set = set()
        self._build_merge_map()
        
        logger.info(f"TranslationSubphase initialized: "
                   f"src={source_language}, tgt={target_language}")
    
    def _build_merge_map(self):
        """Build mapping of merged speakers to their master speakers."""
        for spk_id, info in self.speaker_config.items():
            # Check if this speaker is merged into another
            merged_into = info.get('merged_into')
            if merged_into:
                self._merge_map[spk_id] = merged_into
                logger.info(f"Speaker {spk_id} merged into {merged_into}")
            # Check if this speaker has others merged into it (is a master)
            merged_speakers = info.get('merged_speakers', [])
            if merged_speakers:
                self._master_speakers.add(spk_id)
                for merged in merged_speakers:
                    self._merge_map[str(merged)] = spk_id
                    logger.info(f"Speaker {merged} merged into master {spk_id}")
        
        if self._merge_map:
            logger.info(f"Merge map built: {self._merge_map}")
    
    def _get_effective_speaker_id(self, speaker_id: int) -> int:
        """Get the effective speaker ID after applying merges."""
        spk_str = str(speaker_id)
        if spk_str in self._merge_map:
            return int(self._merge_map[spk_str])
        return speaker_id
    
    def _get_effective_action(self, speaker_id: int) -> str:
        """Get the effective action for a speaker after applying merges."""
        effective_id = self._get_effective_speaker_id(speaker_id)
        effective_str = str(effective_id)
        
        if effective_str in self.speaker_config:
            return self.speaker_config[effective_str].get('action', 'dub')
        return 'dub'  # Default to dub if not found
    
    def _get_effective_speaker_name(self, speaker_id: int) -> str:
        """Get the effective speaker name after applying merges."""
        effective_id = self._get_effective_speaker_id(speaker_id)
        effective_str = str(effective_id)
        
        if effective_str in self.speaker_config:
            return self.speaker_config[effective_str].get('name', f"Speaker {effective_id + 1}")
        
        # Fallback: check original speaker
        spk_str = str(speaker_id)
        if spk_str in self.speaker_config:
            return self.speaker_config[spk_str].get('name', f"Speaker {speaker_id + 1}")
        
        return f"Speaker {speaker_id + 1}"

    async def run(self, segments: List[TranslationSegment]) -> List[TranslationSegment]:
        """
        Translate all segments that need it.
        
        Args:
            segments: List of TranslationSegment objects with original_text
        
        Returns:
            Updated segments with translated_text populated
        """
        # Check if we should skip translation (same language)
        if self._should_skip_translation():
            logger.info(f"Source ({self.source_language}) equals target "
                       f"({self.target_language}), copying original text")
            for ts in segments:
                if ts.status != 'failed' and ts.original_text:
                    ts.translated_text = ts.original_text
                    ts.status = 'synthesizing'
                    # Update speaker name to effective name
                    ts.speaker_name = self._get_effective_speaker_name(ts.speaker_id)
                    
                    # CONSOLE OUTPUT: Show copy (same language)
                    print(f"\n{'='*60}")
                    print(f"[SEGMENT {ts.idx}] TRANSLATION ({self.target_language}):")
                    print(f"  Time: {ts.start:.2f}s - {ts.end:.2f}s | "
                          f"Speaker: {ts.speaker_name}")
                    print(f"  Original: \"{ts.original_text}\"")
                    print(f"  Translated: [Same language - copied]")
                    print(f"{'='*60}\n")
                    
                    await self._log(f"Segment {ts.idx}: copied original text "
                                   f"(same language)")
            
            # Still broadcast progress even if skipping translation
            if self.translation_update_callback:
                await self._broadcast_translation_update(segments)
            
            return segments
        
        # Load Lollms for actual translation
        await self._report_progress("translating", 40, 
            f"Loading translation model for {len(segments)} segments...")
        
        try:
            await self._load_lollms()
            
            if not self._lollms:
                logger.warning("Lollms not available, copying original text")
                for ts in segments:
                    if ts.status != 'failed':
                        ts.translated_text = ts.original_text
                        ts.status = 'synthesizing'
                        ts.speaker_name = self._get_effective_speaker_name(ts.speaker_id)
                return segments
            
            total = len(segments)
            for i, ts in enumerate(segments):
                if ts.status == 'failed' or not ts.original_text:
                    continue
                
                try:
                    # Update speaker name to effective name (after merge resolution)
                    ts.speaker_name = self._get_effective_speaker_name(ts.speaker_id)
                    
                    # Get effective action after merge resolution
                    effective_action = self._get_effective_action(ts.speaker_id)
                    
                    # Check if effective action is 'remove'
                    if effective_action == 'remove':
                        ts.translated_text = ""  # Will result in silence
                        ts.status = 'synthesizing'
                        
                        # CONSOLE OUTPUT: Show removal
                        print(f"\n{'='*60}")
                        print(f"[SEGMENT {ts.idx}] TRANSLATION ({self.target_language}):")
                        print(f"  Time: {ts.start:.2f}s - {ts.end:.2f}s | "
                              f"Speaker: {ts.speaker_name}")
                        print(f"  Original: \"{ts.original_text}\"")
                        print(f"  Translated: [SPEAKER REMOVED - will be silent]")
                        print(f"  Reason: effective_action='remove' for speaker {ts.speaker_id}")
                        print(f"{'='*60}\n")
                        
                        await self._log(f"Segment {ts.idx}: "
                                       f"speaker set to remove, skipping translation")
                        continue
                    
                    # Build translation prompt with explicit target language
                    prompt = self._build_translation_prompt(ts.original_text)
                    
                    # Generate with timeout
                    translated = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._lollms.generate_text,
                            prompt,
                            temperature=0.3
                        ),
                        timeout=30.0
                    )
                    
                    ts.translated_text = translated.strip()
                    ts.status = 'synthesizing'
                    
                    # CONSOLE OUTPUT: Show translation
                    print(f"\n{'='*60}")
                    print(f"[SEGMENT {ts.idx}] TRANSLATION ({self.target_language}):")
                    print(f"  Time: {ts.start:.2f}s - {ts.end:.2f}s | "
                          f"Speaker: {ts.speaker_name}")
                    print(f"  Original: \"{ts.original_text}\"")
                    print(f"  Translated: \"{ts.translated_text}\"")
                    print(f"{'='*60}\n")
                    
                    await self._log(f"Segment {ts.idx} translated to "
                                   f"{self.target_language}: "
                                   f"'{ts.translated_text[:50]}...'")
                    
                    # Progress update
                    progress = 40 + int((i + 1) / total * 20)  # 40-60% range
                    await self._report_progress("translating", progress, 
                        f"Translated {i+1}/{total} segments to {self.target_language}")
                    
                    # Broadcast progressive update every few segments
                    if self.translation_update_callback and (i % 3 == 0 or i == total - 1):
                        await self._broadcast_translation_update(segments[:i+1])
                    
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Translation failed for segment {ts.idx} "
                                f"with traceback:\n{tb_str}")
                    # Fallback: use original text
                    ts.translated_text = ts.original_text
                    ts.status = 'synthesizing'
                    ts.speaker_name = self._get_effective_speaker_name(ts.speaker_id)
                    await self._log(f"Segment {ts.idx}: "
                                   f"translation failed, using original")
            
            # Final broadcast with all segments
            if self.translation_update_callback:
                await self._broadcast_translation_update(segments)
            
            return segments
            
        finally:
            # ALWAYS unload Lollms to free VRAM for TTS
            self._unload_lollms()
    
    def _should_skip_translation(self) -> bool:
        """Check if source equals target (no translation needed)."""
        # Normalize language codes for comparison
        src = (self.source_language or '').lower().strip()
        tgt = (self.target_language or '').lower().strip()
        
        # Direct match
        if src == tgt:
            return True
        
        # Common aliases
        aliases = {
            'en': ['english', 'eng'],
            'es': ['spanish', 'spa', 'espanol', 'español'],
            'fr': ['french', 'fra', 'français'],
            'de': ['german', 'deu', 'deutsch'],
            'it': ['italian', 'ita', 'italiano'],
            'pt': ['portuguese', 'por', 'português'],
            'zh': ['chinese', 'chi', 'zho', '中文'],
            'ja': ['japanese', 'jpn', '日本語'],
            'ko': ['korean', 'kor', '한국어'],
            'ar': ['arabic', 'ara', 'العربية'],
            'ru': ['russian', 'rus', 'русский'],
            'hi': ['hindi', 'hin', 'हिन्दी'],
        }
        
        # Check if src is an alias of tgt or vice versa
        for code, names in aliases.items():
            all_names = [code] + names
            if src in all_names and tgt in all_names:
                return True
        
        return False
    
    async def _load_lollms(self):
        """Load Lollms client."""
        if self._loaded:
            return
        
        logger.info("Loading Lollms translation model...")
        try:
            self._lollms = manager.get_lollms_client()
            self._loaded = True
            logger.info("Lollms loaded")
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Failed to load Lollms with traceback:\n{tb_str}")
            self._lollms = None
            self._loaded = False
    
    def _unload_lollms(self):
        """Unload Lollms to free VRAM."""
        if not self._loaded:
            return
        
        logger.info("Unloading Lollms to free VRAM...")
        self._lollms = None
        self._loaded = False
        manager.clear_cache(keep=['speaker_encoder'])
        import gc
        gc.collect()
        import torch        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Lollms unloaded")
    
    def _build_translation_prompt(self, text: str) -> str:
        """Build translation prompt for Lollms."""
        # Map language codes to full names for better quality
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese', 
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'ru': 'Russian',
            'hi': 'Hindi', 'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish',
            'vi': 'Vietnamese', 'th': 'Thai', 'auto': 'the target language'
        }
        
        lang_name = lang_names.get(self.target_language, self.target_language)
        
        return f"""Translate the following text to {lang_name}. 
Preserve the meaning, tone, and style. Only output the translation, no explanations.

Text: {text}

Translation to {lang_name}:"""
    
    async def _broadcast_translation_update(self, segments: List[TranslationSegment]):
        """Broadcast translation progress to UI."""
        if not self.translation_update_callback:
            return
        
        segments_data = [
            {
                "idx": ts.idx,
                "segment_idx": ts.idx,
                "start": ts.start,
                "end": ts.end,
                "speaker_id": ts.speaker_id,
                "original_text": ts.original_text,
                "translated_text": ts.translated_text if ts.translated_text else "",
                "status": "translated" if ts.translated_text else ts.status
            }
            for ts in segments
        ]
        
        try:
            await self.translation_update_callback(segments_data)
        except Exception as e:
            logger.warning(f"Translation update callback failed: {e}")
    
    async def _report_progress(self, phase: str, percent: int, message: str):
        """Report progress via callback."""
        if self.progress_callback:
            try:
                await self.progress_callback(phase, percent, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def _log(self, message: str):
        """Log message."""
        logger.info(message)


async def run_translation_subphase(
    task_id: str,
    segments: List[TranslationSegment],
    target_language: str = "en",
    source_language: str = "auto",
    speaker_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    translation_update_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None
) -> List[TranslationSegment]:
    """
    Convenience function to run translation subphase.
    
    Args:
        task_id: The task ID
        segments: Segments with original_text to translate
        target_language: Target language code
        source_language: Source language code
        speaker_config: Speaker configuration dict
        progress_callback: Optional callback for progress updates
        translation_update_callback: Optional callback for progressive UI updates
    
    Returns:
        Updated segments with translated_text populated
    """
    subphase = TranslationSubphase(
        task_id=task_id,
        target_language=target_language,
        source_language=source_language,
        speaker_config=speaker_config,
        progress_callback=progress_callback,
        translation_update_callback=translation_update_callback
    )
    
    return await subphase.run(segments)
