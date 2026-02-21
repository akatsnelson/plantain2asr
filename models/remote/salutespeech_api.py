"""
SaluteSpeech REST API client.
Async speech recognition via Sber SmartSpeech.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import uuid
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Tuple


class SaluteSpeechAPI:
    """Client for SaluteSpeech async recognition API."""

    def __init__(self, authorization_key: str, scope: str = "SALUTE_SPEECH_PERS"):
        """
        Args:
            authorization_key: Base64-encoded key from SberDevices Studio.
            scope: ``SALUTE_SPEECH_PERS`` (personal) or ``SALUTE_SPEECH_CORP`` (corporate).
        """
        self.authorization_key = authorization_key
        self.scope = scope
        self.base_url = "https://smartspeech.sber.ru"
        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        self.access_token: Optional[str] = None
        self.token_expires_at: int = 0

        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _generate_rquid(self) -> str:
        return str(uuid.uuid4())

    def _convert_to_mp3(self, audio_path: str) -> str:
        """Convert audio to mono 16 kHz MP3 using ffmpeg (required by the API)."""
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(temp_fd)
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-b:a", "64k",
            "-loglevel", "error", temp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        return temp_path

    def _get_access_token(self) -> str:
        current_time = int(time.time() * 1000)
        if self.access_token and current_time < (self.token_expires_at - 60_000):
            return self.access_token

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": self._generate_rquid(),
            "Authorization": f"Basic {self.authorization_key}",
        }
        response = self.session.post(
            self.oauth_url, headers=headers,
            data={"scope": self.scope}, verify=False, timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        self.access_token = result["access_token"]
        self.token_expires_at = result["expires_at"]
        return self.access_token

    def upload_audio(self, audio_path: str) -> str:
        token = self._get_access_token()
        with open(audio_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/rest/v1/data:upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": f},
                verify=False,
                timeout=60,
            )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != 200:
            raise RuntimeError(f"Upload failed: {result}")
        return result["result"]["request_file_id"]

    def create_recognition_task(self, request_file_id: str, model: str = "general") -> str:
        token = self._get_access_token()
        payload = {
            "request_file_id": request_file_id,
            "options": {"model": model, "audio_encoding": "MP3"},
        }
        response = self.session.post(
            f"{self.base_url}/rest/v1/speech:async_recognize",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            verify=False,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != 200:
            raise RuntimeError(f"Recognition task creation failed: {result}")
        return result["result"]["id"]

    def check_task_status(self, task_id: str) -> Tuple[str, Optional[str]]:
        token = self._get_access_token()
        response = self.session.get(
            f"{self.base_url}/rest/v1/task:get",
            headers={"Authorization": f"Bearer {token}"},
            params={"id": task_id},
            verify=False,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != 200:
            raise RuntimeError(f"Status check failed: {result}")
        task_result = result["result"]
        return task_result["status"], task_result.get("response_file_id")

    def download_result(self, response_file_id: str) -> Dict:
        token = self._get_access_token()
        response = self.session.get(
            f"{self.base_url}/rest/v1/data:download",
            headers={"Authorization": f"Bearer {token}"},
            params={"response_file_id": response_file_id},
            verify=False,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _extract_transcript(self, result) -> str:
        texts = []
        if isinstance(result, list):
            for segment in result:
                for r in segment.get("results", []):
                    text = r.get("normalized_text") or r.get("text", "")
                    if text:
                        texts.append(text)
        return " ".join(texts).strip().rstrip(",").strip()

    def transcribe_async(
        self,
        audio_path: str,
        model: str = "general",
        max_wait_seconds: int = 300,
        poll_interval: int = 2,
        verbose: bool = True,
    ) -> str:
        """
        Full async recognition cycle: upload → create task → poll → download.

        Args:
            audio_path: Path to the audio file (WAV is auto-converted to MP3).
            model: ``general`` (≥8 kHz) or ``callcenter`` (8 kHz telephony).
            max_wait_seconds: Timeout for polling.
            poll_interval: Polling interval in seconds.
            verbose: Print progress messages.

        Returns:
            Recognised transcript string.
        """
        temp_mp3_path = None
        try:
            audio_to_upload = audio_path
            if audio_path.lower().endswith(".wav"):
                if verbose:
                    print("🔄 Converting WAV → MP3...")
                temp_mp3_path = self._convert_to_mp3(audio_path)
                audio_to_upload = temp_mp3_path

            if verbose:
                print(f"📤 Uploading {Path(audio_path).name}...")
            request_file_id = self.upload_audio(audio_to_upload)

            if verbose:
                print("🎙️  Creating recognition task...")
            task_id = self.create_recognition_task(request_file_id, model=model)

            if verbose:
                print(f"⏳ Waiting for result (task_id: {task_id})...")
            start = time.time()
            while True:
                if time.time() - start > max_wait_seconds:
                    raise TimeoutError(f"Recognition timed out after {max_wait_seconds}s")
                status, response_file_id = self.check_task_status(task_id)
                if status == "DONE":
                    if verbose:
                        print("✅ Done!")
                    break
                elif status == "ERROR":
                    raise RuntimeError("Recognition failed with status=ERROR")
                time.sleep(poll_interval)

            result = self.download_result(response_file_id)
            return self._extract_transcript(result)
        finally:
            if temp_mp3_path and os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)
