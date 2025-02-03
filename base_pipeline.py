from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from copy import deepcopy

# List of video databases for face recognition
# https://github.com/polarisZhao/awesome-face/blob/master/README.md

# YouTube Faces DB
# https://www.cs.tau.ac.il//~wolf/ytfaces/

# Design Pattern - Chain of Responsibility
# https://refactoring.guru/design-patterns/chain-of-responsibility/python/example


class NameRequestResponse:
    """
    Class to store the request and response entities
    """

    def __init__(self, name: str, request: dict, response: dict) -> None:
        self.name = name
        self.request = request
        self.response = response

        request_log_key = self.request.get(BaseHandler.log_key)
        if isinstance(request_log_key, str):
            self.request_str = request_log_key
        else:
            self.request_str = f"Request has no [{BaseHandler.log_key}] key"

        response_log_key = self.response.get(BaseHandler.log_key)
        if isinstance(response_log_key, str):
            self.response_str = response_log_key
        else:
            self.response_str = f"Response has no [{BaseHandler.log_key}] key"

    def __str__(self) -> str:
        n_cols = 30
        l1_col2 = self.name  # .ljust(int(n_cols * 2))

        l2_col2 = ", ".join(self.request.keys())
        l2_col2 = f"[{l2_col2}]"  # .ljust(int(n_cols))
        l3_col2 = self.request_str  # .ljust(int(n_cols * 4))

        l4_col2 = ", ".join(self.response.keys())
        l4_col2 = f"[{l4_col2}]"  # .ljust(int(n_cols))
        l5_col2 = self.response_str  # .ljust(int(n_cols * 4))

        l1_col1 = "name:".ljust(n_cols)
        l2_col1 = "  request keys:".ljust(n_cols)
        l3_col1 = f"  request[{BaseHandler.log_key}]:".ljust(n_cols)
        l4_col1 = "  response keys: ".ljust(n_cols)
        l5_col1 = f"  response[{BaseHandler.log_key}]:".ljust(n_cols)

        msg = f"{l1_col1}{l1_col2}\n{l2_col1}{l2_col2}\n{l3_col1}{l3_col2}\n{l4_col1}{l4_col2}\n{l5_col1}{l5_col2}"

        # msg_dict = {
        #     "name": self.name,
        #     "request_keys": l2_col2,
        #     f"request[{BaseHandler.log_key}]": l3_col2,
        #     "response_keys": l4_col2,
        #     f"response[{BaseHandler.log_key}]": l5_col2,
        # }
        # return str(msg_dict)

        return msg


class Handler(ABC):
    """
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.
    """

    log_key = "to_str"
    early_stop_key = "early_stop"

    @abstractmethod
    def set_next(self, handler: Handler) -> Handler:
        """
        Method for building the chain of handlers.
        """

    @abstractmethod
    def handle(self, request) -> NameRequestResponse:
        """
        Method for executing the request
        """

    @abstractmethod
    def get_response(self, request) -> dict:
        """
        Method for processing the request and getting a response
        """

    @abstractmethod
    def early_stop_response(self) -> dict:
        """
        Method for processing the response upon an early stop
        """


class BaseHandler(Handler):
    """
    The default chaining behavior can be implemented inside a base handler
    class.
    """

    def __init__(self) -> None:
        self.__next_handler: Optional[Handler] = None

    def set_next(self, handler: Handler) -> Handler:
        self.__next_handler = handler
        # Returning a handler from here will let us link handlers in a
        # convenient way like this:
        # monkey.set_next(squirrel).set_next(dog)
        return handler

    def handle(self, request: dict) -> NameRequestResponse:
        response = self.get_response(request)
        nrr = NameRequestResponse(
            name=self.__class__.__name__, request=request, response=response
        )
        print(nrr)
        if response.get(Handler.early_stop_key):
            return nrr
        else:
            if hasattr(self.__next_handler, "handle"):
                # return self.__next_handler.handle(request=nrr.response)
                next_handle = getattr(self.__next_handler, "handle")
                return next_handle(nrr.response)
            else:
                return nrr

    def get_response(self, request: dict) -> dict:
        # Add info, then pass it on to the next handler
        # return request
        response = deepcopy(request)
        response[Handler.log_key] = f"Default log from {self.__class__.__name__}"
        return response

    def early_stop_response(self) -> dict:
        return {
            BaseHandler.log_key: "Early stop was triggered",
            BaseHandler.early_stop_key: True,
        }
