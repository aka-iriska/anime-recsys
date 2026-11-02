def require(condition: bool, message: str) -> None:
    """Бросает ValueError с сообщением message, если condition ложно.

    :param condition: Проверяемое условие.
    :param message: Текст ошибки.
    """
    if not condition:
        raise ValueError(message)
