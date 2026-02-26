def seconds_to_timestamp(seconds: int):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:02}"


def timestamp_to_seconds(timestamp: str):
    hrs, mins, secs = map(int, timestamp.split(":"))
    return hrs * 3600 + mins * 60 + secs