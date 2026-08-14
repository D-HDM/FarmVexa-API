# utils/gemini_key_fallback.py

from config.settings import settings


def is_key_limit_error(error):
    """Check if error is related to API key limits/quota."""
    error_str = str(error).lower()
    return any([
        "429" in error_str,
        "403" in error_str,
        "quota" in error_str,
        "exceeded" in error_str,
        "rate limit" in error_str,
        "resource exhausted" in error_str,
    ])


def get_key_pair(service="crop"):
    """
    Return (primary_key, backup_key) for the given service.
    
    Services:
    - "crop" or "chat" → GEMINI_API_KEY + GEMINI_API_KEY_BACKUP
    - "fieldscan" → GEMINI_FIELDSCAN_API_KEY + GEMINI_FIELDSCAN_API_KEY_BACKUP
    """
    if service == "fieldscan":
        return (
            settings.GEMINI_FIELDSCAN_API_KEY,
            settings.GEMINI_FIELDSCAN_API_KEY_BACKUP,
        )
    else:
        return (
            settings.GEMINI_API_KEY,
            settings.GEMINI_API_KEY_BACKUP,
        )


def has_backup(service="crop"):
    """Check if the service has a backup key configured."""
    _, backup = get_key_pair(service)
    return bool(backup)