def __getattr__(name):
    if name == "DataProcessor":
        from processing.data_processor import DataProcessor
        return DataProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DataProcessor"]
