from datetime import datetime


def success_response(data: dict, message: str = "Success"):
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


def error_response(message: str, error_code: int = 400):
    return {
        "success": False,
        "message": message,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat()
    }