"""
backend/services/file_storage.py
─────────────────────────────────────────────────────────────
Secure file storage and upload validation service.

Key capabilities:
  - Filename sanitization against path traversal
  - Allowed extension and file signature (magic bytes) validation
  - Streaming in 64 KB chunks to prevent memory exhaustion
  - Enforcing 50 MB maximum size cap
  - Rejection of empty files
  - SHA-256 calculation on the fly
  - Storing under data/uploads/{collection_id}/ with UUID-prefixed names
  - Automatic cleanup of partial files on failure
  - Boundary check: never modify or delete files outside upload_dir
─────────────────────────────────────────────────────────────
"""

import hashlib
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from backend.config import get_settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks

# Magic bytes for supported file formats
PDF_SIGNATURE = b"%PDF-"
DOCX_SIGNATURE = b"PK\x03\x04"


class StorageError(Exception):
    """Base class for storage errors."""
    pass


class InvalidFileError(StorageError):
    """Raised when file extension or magic bytes are unsupported."""
    pass


class EmptyFileError(StorageError):
    """Raised when uploaded file has 0 bytes."""
    pass


class FileTooLargeError(StorageError):
    """Raised when uploaded file exceeds max_upload_size_mb."""
    pass


class DirectoryTraversalError(StorageError):
    """Raised if an attempted path escapes the upload boundary."""
    pass


@dataclass
class StoredFileInfo:
    original_filename: str
    stored_filename: str
    file_type: str
    file_path: Path
    size_bytes: int
    sha256: str


class FileStorageService:
    def __init__(self, upload_dir: Optional[Path] = None):
        settings = get_settings()
        self.upload_dir = (upload_dir or settings.upload_dir_path).resolve()
        self.max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        self.supported_extensions = set(settings.supported_extensions_list)
        # Ensure base upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: Optional[str]) -> str:
        """
        Sanitize filename to prevent directory traversal and invalid characters.
        Strips path separators and restricts to safe alphanumeric, dot, underscore, and dash.
        """
        if not filename:
            return f"upload_{uuid.uuid4().hex[:8]}"

        # Take only the base name
        base_name = Path(filename).name
        # Replace dangerous or whitespace characters
        sanitized = re.sub(r"[^\w.\-]", "_", base_name)
        # Remove consecutive dots to avoid hidden directory traversal
        sanitized = re.sub(r"\.{2,}", ".", sanitized).strip("._")
        if not sanitized:
            sanitized = f"upload_{uuid.uuid4().hex[:8]}"
        return sanitized

    def _verify_boundary(self, path: Path) -> Path:
        """Ensure the target path is strictly within the allowed upload_dir."""
        resolved = path.resolve()
        try:
            resolved.relative_to(self.upload_dir)
        except ValueError:
            raise DirectoryTraversalError(
                f"Security violation: path {resolved} is outside upload directory {self.upload_dir}"
            )
        return resolved

    def _validate_signature(self, ext: str, first_bytes: bytes) -> None:
        """Validate header signatures for binary types."""
        if ext == ".pdf":
            if not first_bytes.startswith(PDF_SIGNATURE):
                raise InvalidFileError(
                    "File signature mismatch: expected valid PDF header (%PDF-)"
                )
        elif ext == ".docx":
            if not first_bytes.startswith(DOCX_SIGNATURE):
                raise InvalidFileError(
                    "File signature mismatch: expected valid DOCX zip header (PK\x03\x04)"
                )
        elif ext == ".txt":
            # Reject text files with null bytes (binary content disguised as txt)
            if b"\x00" in first_bytes:
                raise InvalidFileError(
                    "Invalid text file: binary null bytes detected in TXT file"
                )

    async def save_uploaded_file(
        self,
        collection_id: str,
        upload_file: UploadFile,
    ) -> StoredFileInfo:
        """
        Stream, validate, hash, and persist an uploaded file.
        Cleans up any partially written files if an error occurs.
        """
        original_filename = self.sanitize_filename(upload_file.filename)
        ext = Path(original_filename).suffix.lower()

        if ext not in self.supported_extensions:
            raise InvalidFileError(
                f"Unsupported file type '{ext}'. Supported formats: {sorted(self.supported_extensions)}"
            )

        file_type = ext.lstrip(".")
        collection_dir = self._verify_boundary(self.upload_dir / collection_id)
        collection_dir.mkdir(parents=True, exist_ok=True)

        stored_filename = f"{uuid.uuid4().hex}_{original_filename}"
        target_path = self._verify_boundary(collection_dir / stored_filename)

        hasher = hashlib.sha256()
        total_size = 0
        first_chunk = True

        try:
            with open(target_path, "wb") as out_file:
                while True:
                    chunk = await upload_file.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    if first_chunk:
                        if len(chunk) == 0:
                            raise EmptyFileError("Uploaded file is empty (0 bytes)")
                        self._validate_signature(ext, chunk[:16])
                        first_chunk = False

                    total_size += len(chunk)
                    if total_size > self.max_size_bytes:
                        raise FileTooLargeError(
                            f"File size exceeds maximum allowed limit of {self.max_size_bytes // (1024*1024)} MB"
                        )

                    hasher.update(chunk)
                    out_file.write(chunk)

            if total_size == 0:
                raise EmptyFileError("Uploaded file is empty (0 bytes)")

        except Exception:
            # Clean up partial file on any failure
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError as err:
                    logger.warning("Failed to delete partial file %s: %s", target_path, err)
            raise

        return StoredFileInfo(
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_type=file_type,
            file_path=target_path,
            size_bytes=total_size,
            sha256=hasher.hexdigest(),
        )

    def get_stored_file_path(self, collection_id: str, stored_filename: str) -> Path:
        """
        Securely resolve the path to a stored file on disk.
        Enforces upload boundary check to prevent path traversal and asserts file exists.
        
        Raises:
            DirectoryTraversalError: if resolved path escapes upload_dir.
            FileNotFoundError: if target file does not exist on disk.
        """
        target_path = self.upload_dir / collection_id / stored_filename
        resolved = self._verify_boundary(target_path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Stored file not found: {stored_filename} in collection {collection_id}"
            )
        return resolved

    def delete_stored_file(self, collection_id: str, stored_filename: str) -> bool:
        """
        Safely delete a stored file from disk.
        Returns True if the file was deleted or is already missing (already cleaned).
        Returns False on permission/OSError or directory traversal violation.
        """
        target_path = self.upload_dir / collection_id / stored_filename
        try:
            resolved = self._verify_boundary(target_path)
            if resolved.is_file():
                resolved.unlink()
                logger.info("Deleted stored document file: %s", resolved)
                return True
            if not resolved.exists():
                logger.info("Stored document file already absent: %s", resolved)
                return True
            return False
        except DirectoryTraversalError:
            logger.error("Attempted deletion outside upload dir: %s", target_path)
            return False
        except OSError as e:
            logger.error("Failed to delete file %s: %s", target_path, e)
            return False

    def delete_collection_storage(self, collection_id: str) -> bool:
        """
        Safely delete an entire collection's storage directory.
        Returns True if the directory was deleted or is already missing (already cleaned).
        Returns False on permission/OSError or directory traversal violation.
        """
        target_dir = self.upload_dir / collection_id
        try:
            resolved = self._verify_boundary(target_dir)
            if resolved.is_dir():
                def _handle_readonly(func, path, exc):
                    import errno
                    import stat

                    # Unwrap exc if passed as exc_info tuple (Python < 3.12 compatibility)
                    exc_val = exc[1] if isinstance(exc, tuple) else exc

                    # Handle read-only file attributes specifically on PermissionError
                    if isinstance(exc_val, PermissionError) or getattr(exc_val, "errno", None) in (
                        errno.EACCES,
                        errno.EPERM,
                    ):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    else:
                        raise exc_val

                try:
                    shutil.rmtree(resolved, onexc=_handle_readonly)
                except TypeError:
                    shutil.rmtree(resolved, onerror=_handle_readonly)

                # Verify target directory no longer exists before returning True
                if not resolved.exists():
                    logger.info("Deleted collection storage directory: %s", resolved)
                    return True
                logger.error("Collection storage directory still exists after rmtree: %s", resolved)
                return False
            if not resolved.exists():
                logger.info("Collection storage directory already absent: %s", resolved)
                return True
            return False
        except DirectoryTraversalError:
            logger.error("Attempted rmtree outside upload dir: %s", target_dir)
            return False
        except OSError as e:
            logger.error("Failed to delete collection dir %s: %s", target_dir, e)
            return False
