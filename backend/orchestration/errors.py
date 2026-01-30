class PipelineError(Exception):
    def __init__(self, error_code: str, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.trace_id = trace_id
