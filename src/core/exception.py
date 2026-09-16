import sys


def error_message_detail(error: str, error_detail: sys) -> str:
    """
    Extracts detailed information about an exception,
    including the filename and line number where the error
    occurred.
    Args:
        error: The exception object.
        error_detail: The sys module to access exception
                      information.
    Returns:
        A formatted string containing the filename, line number,
        and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in script: [{file_name}], "
        f"at line number: [{exc_tb.tb_lineno}], "
        f"error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    A custom exception class that extends the built-in
    Exception class. It provides detailed error messages
    for exceptions raised in the code.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the CustomException with an error message
        and detailed information about the exception.

        Args:
            error_message: The error message to be associated
                           with the exception.
            error_detail: The sys module to access exception
                          information.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the CustomException.
        """
        return self.error_message
