from base_pipeline import BaseHandler, Handler

# List of video databases for face recognition
# https://github.com/polarisZhao/awesome-face/blob/master/README.md

# YouTube Faces DB
# https://www.cs.tau.ac.il//~wolf/ytfaces/

# Design Pattern - Chain of Responsibility
# https://refactoring.guru/design-patterns/chain-of-responsibility/python/example


class MonkeyHandler(BaseHandler):
    def get_response(self, request: dict) -> dict:
        if request[BaseHandler.log_key] == "Banana":
            response = f"{self.__class__.__name__}: I'll eat the {request[BaseHandler.log_key]}"
            return {BaseHandler.log_key: response}
        else:
            # return super().get_response(request)
            return super().early_stop_response()


class SquirrelHandler(BaseHandler):
    def get_response(self, request: dict) -> dict:
        if request[BaseHandler.log_key] == "Nut":
            response = f"{self.__class__.__name__}: I'll eat the {request[BaseHandler.log_key]}"
            return {BaseHandler.log_key: response}
        else:
            # return super().get_response(request)
            return super().early_stop_response()


class DogHandler(BaseHandler):
    def get_response(self, request: dict) -> dict:
        if request[BaseHandler.log_key] == "MeatBall":
            response = f"{self.__class__.__name__}: I'll eat the {request[BaseHandler.log_key]}"
            return {BaseHandler.log_key: response}
        else:
            # return super().get_response(request)
            return super().early_stop_response()


def client_code(handler: Handler) -> None:
    """
    The client code is usually suited to work with a single handler. In most
    cases, it is not even aware that the handler is part of a chain.
    """

    for log_key_request in ["Nut", "Banana", "Cup of coffee"]:
        request = {BaseHandler.log_key: log_key_request}
        print()
        print(
            f"Client input request[{BaseHandler.log_key}]: {request[BaseHandler.log_key]}"
        )
        nrr = handler.handle(request)
        print(
            f"Client output response[{BaseHandler.log_key}]: {nrr.response[BaseHandler.log_key]}"
        )
        # if result:
        #     print(f"  {result}", end="")
        # else:
        #     print(f"  {food} was left untouched.", end="")


if __name__ == "__main__":
    monkey = MonkeyHandler()
    squirrel = SquirrelHandler()
    dog = DogHandler()

    monkey.set_next(squirrel).set_next(dog)

    # The client should be able to send a request to any handler, not just the
    # first one in the chain.
    print("Chain: Monkey > Squirrel > Dog")
    client_code(monkey)
    print("\n")

    # print("Subchain: Squirrel > Dog")
    # client_code(squirrel)
